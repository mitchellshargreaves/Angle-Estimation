close all; clear all; clc;
sim_t = 0.6; % length of simulation
dt = 5e-5; % timestep

num_sims = 1024;
num_loads = 10; % number of loads

% Set RL
R_val = 1;
L_val = 1e-2;

R = ones(num_sims, num_loads) * R_val;
L = ones(num_sims, num_loads) * L_val;

% Randomise P
P_min = 500;
P_max = 5000;
P = rand(num_sims, num_loads) * (P_max - P_min) + P_min;


% Initialise Simulations
sim_sample = Simulink.SimulationInput('Y_Model_2_Loads');
sim_sample = sim_sample.setVariable('sim_t',sim_t);
sim_sample = sim_sample.setVariable('dt',dt);
in = repmat(sim_sample, 1, num_sims);

% Add the simulation parameters 
for i = 1 : num_sims
   in(i) = in(i).setVariable('R', R(i, :));
   in(i) = in(i).setVariable('P', P(i, :));
   in(i) = in(i).setVariable('L', L(i, :));
end

% Run simulation
out = parsim(in, 'ShowProgress', 'on', 'ShowSimulationManager', 'on', 'TransferBaseWorkspaceVariables', 'on', 'StopOnError', 'on');

% Visualise simulations
x = 0 : .02 : sim_t - .02;

% Paths
metadatapath = "2_load_metadata/";
scadadatapath = "2_load_scadadata/";
datapath = "2_load_data/";

offset = 0;

for i = 1 : num_sims
    % Record metadata
    Rs = R(i, :)';
    Ps = P(i, :)';
    Ls = L(i, :)';
    
    metadata = table(Rs, Ps, Ls);
    writetable(metadata, metadatapath + string(offset + i) + ".csv")
    
    % Record scata data
    scada_mag1 = out(i).scada_mag1(2:end);
    scada_ang1 = out(i).scada_ang1(2:end);

    scada_mag2 = out(i).scada_mag2(2:end);
    scada_ang2 = out(i).scada_ang2(2:end);

    scada_mag4 = out(i).scada_mag4(2:end);
    scada_ang4 = out(i).scada_ang4(2:end);

%     scada_mag5 = out(i).scada_mag5(2:end);
%     scada_ang5 = out(i).scada_ang5(2:end);
% 
%     scada_mag7 = out(i).scada_mag7(2:end);
%     scada_ang7 = out(i).scada_ang7(2:end);
% 
%     scada_mag8 = out(i).scada_mag8(2:end);
%     scada_ang8 = out(i).scada_ang8(2:end);
% 
%     scada_mag10 = out(i).scada_mag10(2:end);
%     scada_ang10 = out(i).scada_ang10(2:end);
% 
%     scada_mag11 = out(i).scada_mag11(2:end);
%     scada_ang11 = out(i).scada_ang11(2:end);
% 
%     scada_mag13 = out(i).scada_mag13(2:end);
%     scada_ang13 = out(i).scada_ang13(2:end);
% 
%     scada_mag14 = out(i).scada_mag14(2:end);
%     scada_ang14 = out(i).scada_ang14(2:end);
% 
%     scada_mag16 = out(i).scada_mag16(2:end);
%     scada_ang16 = out(i).scada_ang16(2:end);
% 
%     scada_mag17 = out(i).scada_mag17(2:end);
%     scada_ang17 = out(i).scada_ang17(2:end);
% 
%     scada_mag19 = out(i).scada_mag19(2:end);
%     scada_ang19 = out(i).scada_ang19(2:end);
% 
%     scada_mag20 = out(i).scada_mag20(2:end);
%     scada_ang20 = out(i).scada_ang20(2:end);
    
%     scatadata = table(scada_mag1, scada_mag2, scada_mag4, scada_mag5, scada_mag7, scada_mag8, scada_mag10, scada_mag11, scada_mag13, scada_mag14, scada_mag16, scada_mag17, scada_mag19, scada_mag20, scada_ang1, scada_ang2, scada_ang4, scada_ang5, scada_ang7, scada_ang8, scada_ang10, scada_ang11, scada_ang13, scada_ang14, scada_ang16, scada_ang17, scada_ang19, scada_ang20);
    scatadata = table(scada_mag1, scada_mag2, scada_ang1, scada_ang2);
    writetable(scatadata, scadadatapath + string(offset + i) + ".csv")
    
    % Record data
    i_ang1 = out(i).i_ang1(2:end);
    i_freq1 = out(i).i_freq1(2:end);
    i_mag1 = out(i).i_mag1(2:end);
    v_ang1 = out(i).v_ang1(2:end);
    v_freq1 = out(i).v_freq1(2:end);
    v_mag1 = out(i).v_mag1(2:end);

    i_ang2 = out(i).i_ang2(2:end);
    i_freq2 = out(i).i_freq2(2:end);
    i_mag2 = out(i).i_mag2(2:end);
    v_ang2 = out(i).v_ang2(2:end);
    v_freq2 = out(i).v_freq2(2:end);
    v_mag2 = out(i).v_mag2(2:end);

    i_ang3 = out(i).i_ang3(2:end);
    i_freq3 = out(i).i_freq3(2:end);
    i_mag3 = out(i).i_mag3(2:end);
    v_ang3 = out(i).v_ang3(2:end);
    v_freq3 = out(i).v_freq3(2:end);
    v_mag3 = out(i).v_mag3(2:end);

    i_ang4 = out(i).i_ang4(2:end);
    i_freq4 = out(i).i_freq4(2:end);
    i_mag4 = out(i).i_mag4(2:end);
    v_ang4 = out(i).v_ang4(2:end);
    v_freq4 = out(i).v_freq4(2:end);
    v_mag4 = out(i).v_mag4(2:end);

    i_ang5 = out(i).i_ang5(2:end);
    i_freq5 = out(i).i_freq5(2:end);
    i_mag5 = out(i).i_mag5(2:end);
    v_ang5 = out(i).v_ang5(2:end);
    v_freq5 = out(i).v_freq5(2:end);
    v_mag5 = out(i).v_mag5(2:end);
% 
%     i_ang6 = out(i).i_ang6(2:end);
%     i_freq6 = out(i).i_freq6(2:end);
%     i_mag6 = out(i).i_mag6(2:end);
%     v_ang6 = out(i).v_ang6(2:end);
%     v_freq6 = out(i).v_freq6(2:end);
%     v_mag6 = out(i).v_mag6(2:end);
% 
%     i_ang7 = out(i).i_ang7(2:end);
%     i_freq7 = out(i).i_freq7(2:end);
%     i_mag7 = out(i).i_mag7(2:end);
%     v_ang7 = out(i).v_ang7(2:end);
%     v_freq7 = out(i).v_freq7(2:end);
%     v_mag7 = out(i).v_mag7(2:end);
% 
%     i_ang8 = out(i).i_ang8(2:end);
%     i_freq8 = out(i).i_freq8(2:end);
%     i_mag8 = out(i).i_mag8(2:end);
%     v_ang8 = out(i).v_ang8(2:end);
%     v_freq8 = out(i).v_freq8(2:end);
%     v_mag8 = out(i).v_mag8(2:end);
% 
%     i_ang9 = out(i).i_ang9(2:end);
%     i_freq9 = out(i).i_freq9(2:end);
%     i_mag9 = out(i).i_mag9(2:end);
%     v_ang9 = out(i).v_ang9(2:end);
%     v_freq9 = out(i).v_freq9(2:end);
%     v_mag9 = out(i).v_mag9(2:end);
% 
%     i_ang10 = out(i).i_ang10(2:end);
%     i_freq10 = out(i).i_freq10(2:end);
%     i_mag10 = out(i).i_mag10(2:end);
%     v_ang10 = out(i).v_ang10(2:end);
%     v_freq10 = out(i).v_freq10(2:end);
%     v_mag10 = out(i).v_mag10(2:end);
% 
%     i_ang11 = out(i).i_ang11(2:end);
%     i_freq11 = out(i).i_freq11(2:end);
%     i_mag11 = out(i).i_mag11(2:end);
%     v_ang11 = out(i).v_ang11(2:end);
%     v_freq11 = out(i).v_freq11(2:end);
%     v_mag11 = out(i).v_mag11(2:end);
% 
%     i_ang12 = out(i).i_ang12(2:end);
%     i_freq12 = out(i).i_freq12(2:end);
%     i_mag12 = out(i).i_mag12(2:end);
%     v_ang12 = out(i).v_ang12(2:end);
%     v_freq12 = out(i).v_freq12(2:end);
%     v_mag12 = out(i).v_mag12(2:end);
% 
%     i_ang13 = out(i).i_ang13(2:end);
%     i_freq13 = out(i).i_freq13(2:end);
%     i_mag13 = out(i).i_mag13(2:end);
%     v_ang13 = out(i).v_ang13(2:end);
%     v_freq13 = out(i).v_freq13(2:end);
%     v_mag13 = out(i).v_mag13(2:end);
% 
%     i_ang14 = out(i).i_ang14(2:end);
%     i_freq14 = out(i).i_freq14(2:end);
%     i_mag14 = out(i).i_mag14(2:end);
%     v_ang14 = out(i).v_ang14(2:end);
%     v_freq14 = out(i).v_freq14(2:end);
%     v_mag14 = out(i).v_mag14(2:end);
% 
%     i_ang15 = out(i).i_ang15(2:end);
%     i_freq15 = out(i).i_freq15(2:end);
%     i_mag15 = out(i).i_mag15(2:end);
%     v_ang15 = out(i).v_ang15(2:end);
%     v_freq15 = out(i).v_freq15(2:end);
%     v_mag15 = out(i).v_mag15(2:end);
% 
%     i_ang16 = out(i).i_ang16(2:end);
%     i_freq16 = out(i).i_freq16(2:end);
%     i_mag16 = out(i).i_mag16(2:end);
%     v_ang16 = out(i).v_ang16(2:end);
%     v_freq16 = out(i).v_freq16(2:end);
%     v_mag16 = out(i).v_mag16(2:end);
% 
%     i_ang17 = out(i).i_ang17(2:end);
%     i_freq17 = out(i).i_freq17(2:end);
%     i_mag17 = out(i).i_mag17(2:end);
%     v_ang17 = out(i).v_ang17(2:end);
%     v_freq17 = out(i).v_freq17(2:end);
%     v_mag17 = out(i).v_mag17(2:end);
% 
%     i_ang18 = out(i).i_ang18(2:end);
%     i_freq18 = out(i).i_freq18(2:end);
%     i_mag18 = out(i).i_mag18(2:end);
%     v_ang18 = out(i).v_ang18(2:end);
%     v_freq18 = out(i).v_freq18(2:end);
%     v_mag18 = out(i).v_mag18(2:end);
% 
%     i_ang19 = out(i).i_ang19(2:end);
%     i_freq19 = out(i).i_freq19(2:end);
%     i_mag19 = out(i).i_mag19(2:end);
%     v_ang19 = out(i).v_ang19(2:end);
%     v_freq19 = out(i).v_freq19(2:end);
%     v_mag19 = out(i).v_mag19(2:end);
% 
%     i_ang20 = out(i).i_ang20(2:end);
%     i_freq20 = out(i).i_freq20(2:end);
%     i_mag20 = out(i).i_mag20(2:end);
%     v_ang20 = out(i).v_ang20(2:end);
%     v_freq20 = out(i).v_freq20(2:end);
%     v_mag20 = out(i).v_mag20(2:end);
% 
%     i_ang21 = out(i).i_ang21(2:end);
%     i_freq21 = out(i).i_freq21(2:end);
%     i_mag21 = out(i).i_mag21(2:end);
%     v_ang21 = out(i).v_ang21(2:end);
%     v_freq21 = out(i).v_freq21(2:end);
%     v_mag21 = out(i).v_mag21(2:end);

%     % For debugging only
%     % Plot after
%     figure
%     subplot(6,1,1);
%     plot(x, aft_i_freq)
%     subplot(6,1,2);
%     plot(x, aft_i_mag)
%     subplot(6,1,3);
%     plot(x, aft_i_ang)
%     subplot(6,1,4);
%     plot(x, aft_v_freq)
%     subplot(6,1,5);
%     plot(x, aft_v_mag)
%     subplot(6,1,6);
%     plot(x, aft_v_ang)

%     comb_data = table(i_ang1, i_freq1, i_mag1, v_ang1, v_freq1, v_mag1, i_ang2, i_freq2, i_mag2, v_ang2, v_freq2, v_mag2, i_ang3, i_freq3, i_mag3, v_ang3, v_freq3, v_mag3, i_ang4, i_freq4, i_mag4, v_ang4, v_freq4, v_mag4, i_ang5, i_freq5, i_mag5, v_ang5, v_freq5, v_mag5, i_ang6, i_freq6, i_mag6, v_ang6, v_freq6, v_mag6, i_ang7, i_freq7, i_mag7, v_ang7, v_freq7, v_mag7, i_ang8, i_freq8, i_mag8, v_ang8, v_freq8, v_mag8, i_ang9, i_freq9, i_mag9, v_ang9, v_freq9, v_mag9, i_ang10, i_freq10, i_mag10, v_ang10, v_freq10, v_mag10, i_ang11, i_freq11, i_mag11, v_ang11, v_freq11, v_mag11, i_ang12, i_freq12, i_mag12, v_ang12, v_freq12, v_mag12, i_ang13, i_freq13, i_mag13, v_ang13, v_freq13, v_mag13, i_ang14, i_freq14, i_mag14, v_ang14, v_freq14, v_mag14, i_ang15, i_freq15, i_mag15, v_ang15, v_freq15, v_mag15, i_ang16, i_freq16, i_mag16, v_ang16, v_freq16, v_mag16, i_ang17, i_freq17, i_mag17, v_ang17, v_freq17, v_mag17, i_ang18, i_freq18, i_mag18, v_ang18, v_freq18, v_mag18, i_ang19, i_freq19, i_mag19, v_ang19, v_freq19, v_mag19, i_ang20, i_freq20, i_mag20, v_ang20, v_freq20, v_mag20, i_ang21, i_freq21, i_mag21, v_ang21, v_freq21, v_mag21);
    comb_data = table(i_ang1, i_freq1, i_mag1, v_ang1, v_freq1, v_mag1, i_ang2, i_freq2, i_mag2, v_ang2, v_freq2, v_mag2, i_ang3, i_freq3, i_mag3, v_ang3, v_freq3, v_mag3, i_ang4, i_freq4, i_mag4, v_ang4, v_freq4, v_mag4, i_ang5, i_freq5, i_mag5, v_ang5, v_freq5, v_mag5);
    writetable(comb_data, datapath + string(offset + i) + "_comb.csv")    
end