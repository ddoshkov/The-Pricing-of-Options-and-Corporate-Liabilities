%{
Detta skript analyserar Early Exercise Premium (EEP) för amerikanska
säljoptioner, definierat som den relativa prisskillnaden mellan en
amerikansk och en europeisk put:

EEP = (V_amer - V_eur) / V_eur

Vi undersöker hur EEP beror på moneyness (S/K), löptid (T),
riskfri ränta (r) och volatilitet (sigma) genom att visualisera
alla sex kombinationer av två parametrar som 3D-ytor, med de
övriga två hållna fasta.
%}

clear; clc; close all;

K = 1; % lösenpris, normaliserad till 1
Smax = 3; % Godtycklig övre gräns för aktiepriset. När S>>K är optionen värdelös
N = 100; %diskretisering i S
M = 100; % diskretisering i tid

% Parametervektorer som varieras i ytplottarna
sigma_vec = 0.1:0.05:0.5; % Volatilitet
r_vec = 0.0:0.01:0.10; % Riskfri ränta
T_vec = 0.25:0.25:3.0; % Löptid
SK_vec  = 0.3:0.1:1.0; % Moneyness 

% Fixerade värden för plottarna
r0 = 0.05;
sig0 = 0.2;
T0 = 1.0;
SK0 = 0.7;


%% Plots

% r vs sigma (fix T, S/K)
plot_EEP(r_vec, sigma_vec, ...
    @(r,s) solve_BS(s, r, T0, K, Smax, N, M, SK0), ...
    "r", "\sigma", sprintf("T=%.2f, S/K=%.1f", T0, SK0), 1);

% r vs T (fix sigma, S/K)
plot_EEP(r_vec, T_vec, ...
    @(r,T) solve_BS(sig0, r, T, K, Smax, N, M, SK0), ...
    "r", "T", sprintf("\\sigma=%.2f, S/K=%.1f", sig0, SK0), 2);

% r vs S/K (fix sigma, T)
plot_EEP(r_vec, SK_vec, ...
    @(r,sk) solve_BS(sig0, r, T0, K, Smax, N, M, sk), ...
    "r", "S/K", sprintf("\\sigma=%.2f, T=%.2f", sig0, T0), 3);

% sigma vs T (fix r, S/K)
plot_EEP(sigma_vec, T_vec, ...
    @(s,T) solve_BS(s, r0, T, K, Smax, N, M, SK0), ...
    "\sigma", "T", sprintf("r=%.2f, S/K=%.1f", r0, SK0), 4);

% sigma vs S/K (fix r, T)
plot_EEP(sigma_vec, SK_vec, ...
    @(s,sk) solve_BS(s, r0, T0, K, Smax, N, M, sk), ...
    "\sigma", "S/K", sprintf("r=%.2f, T=%.2f", r0, T0), 5);

% T vs S/K (fix r, sigma)
plot_EEP(T_vec, SK_vec, ...
    @(T,sk) solve_BS(sig0, r0, T, K, Smax, N, M, sk), ...
    "T", "S/K", sprintf("r=%.2f, \\sigma=%.2f", r0, sig0), 6);

%% Funktioner 

% plot_EEP - Plottar EEP som en 3D-yta över två parametrar
% Input:
% xvec - Vektor med värden för x-axelns parameter
% yvec - Vektor med värden för y-axelns parameter
% f - Funktionshandtag f(x,y) som returnerar EEP-värdet
% för given kombination av x och y
% xl - Etikett för x-axeln (sträng)
% yl - Etikett för y-axeln (sträng)
% param_str - Beskrivning av fixerade parametrar, visas i titeln
% num - Figurens nummer

function plot_EEP(xvec, yvec, f, xl, yl, param_str, num)
    
    % Beräkna EEP för alla kombinationer av x och y
    EEP = zeros(length(yvec), length(xvec));
    for yi = 1:length(yvec)
        for xi = 1:length(xvec)
            EEP(yi,xi) = f(xvec(xi), yvec(yi));
        end
    end

     % Skapa rutnät för plot och visualisera
    [X, Y] = meshgrid(xvec, yvec);
    figure(num);
    surf(X, Y, EEP, "EdgeColor", "none");
    xlabel(xl);
    ylabel(yl);
    zlabel("(V_{amer}-V_{eur})/V_{eur}");
    title(sprintf("EEP relativ V_{eur}, %s", param_str));
    colorbar;
    view(45, 30);
end

% solve_BS - Löser Black-Scholes PDE med Crank-Nicolson och returnerar EEP
% Input:
% sigma - Volatilitet
% r - Riskfri ränta
% T - Löptid [år]
% K - Lösenpris
% Smax - Övre gräns för aktiepriset
% N - Antal diskretiseringspunkter i S-led
% M - Antal tidssteg
% SK - Moneyness S/K vid vilken EEP utvärderas
% Output:
% EEP - Relativ prisskillnad (V_amer - V_eur) / V_eur

function EEP = solve_BS(sigma, r, T, K, Smax, N, M, SK)
    
    % Steglängder och rutnät
    dS = Smax / N;
    dt = T / M;
    S = (0:N)' * dS;
    i_eval = max(2, min(N, round(SK*K/dS) + 1)); % Index i S-vektorn 
    
    % Terminalvillkor vid t=T:
    Ve = max(K - S, 0); Va = Ve;
    
    % Crank-Nicolson koefficienter för inre punkter.
    % Härleds från diskretisering av BS-PDE med centrala differenser.
    al = zeros(N+1, 1); % Subdiagonal
    be = zeros(N+1, 1); % Diagonal
    ga = zeros(N+1, 1); % Superdiagonal
    for i = 2:N
        Si = S(i);
        al(i) = dt/4 * (sigma^2*Si^2/dS^2 - r*Si/dS);
        be(i) = dt/2 * (sigma^2*Si^2/dS^2 + r);
        ga(i) = dt/4 * (sigma^2*Si^2/dS^2 + r*Si/dS);
    end
    
    % Sätt ihop de tridiagonala matriserna A och B. 
    A = diag(1 + be(2:N)) - diag(ga(2:N-1), 1) - diag(al(3:N), -1);
    B = diag(1 - be(2:N)) + diag(ga(2:N-1), 1) + diag(al(3:N), -1);
    al2 = al(2); % koefficienten för S=0
    
    % Tidsstegning bakåt från t=T till t=0.
    for j = 1:M
        tau = T - j*dt; % Återstående tid till förfall
        V0 = K * exp(-r * tau); % Randvillkor vid S=0
        
        %{  
        I CN måste man lösa ut nästa steg ur 
        A*V^{n+1} = B*V^n,
        samt ta hänsyn till korrektioner i randen. 
        %}

        % Högerled för europeisk put
        rhs_e = B * Ve(2:N);
        rhs_e(1) = rhs_e(1) + al2 * V0;
        
        % Högerled för amerikansk put
        rhs_a = B * Va(2:N);
        rhs_a(1) = rhs_a(1) + al2 * V0;
        
        % Lös A*V^{n+1} = rhs för inre punkter
        Ve(2:N) = A \ rhs_e;

        % Amerikansk: optionen får ej vara värd mindre än max(K-S, 0)
        Va(2:N) = max(A \ rhs_a, K - S(2:N));
        
        % Sätt randvillkor explicit vid varje tidssteg
        % Europeisk: 
        Ve(1) = V0; % V(0)=K*exp(-rT),
        Ve(N+1) = 0; % V(Smax)=0
        % Amerikansk: 
        Va(1) = K; % V(0)=K 
        Va(N+1) = 0; % V(Smax)=0

    end

    EEP = (Va(i_eval) - Ve(i_eval)) / Ve(i_eval);
end
