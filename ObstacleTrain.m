clear all; close all; clc
%% Defining an environment
Neurons1 = 50;
env = ObstacleEnv;
obsInfo = getObservationInfo(env);
numObservations = obsInfo.Dimension(1);
actInfo = getActionInfo(env);

rng(22);
%% DQN Critic network
statePath = [
    imageInputLayer([3 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(Neurons1,'Name','stateFC1')
    leakyReluLayer(0.001,'Name','relusp') 
    fullyConnectedLayer(Neurons1,'Name','CriticStateFC2')];

commonPath = [
     additionLayer(2,'Name','add')
     leakyReluLayer(0.001,'Name','relucp')
     fullyConnectedLayer(Neurons1,'Name','FCcommon')
     leakyReluLayer(0.001,'Name','relucp2')
     fullyConnectedLayer(1,'Name','ActorOutput')];

actionPath = [
    imageInputLayer([1 1 1],'Normalization','none', 'Name', 'action')
    fullyConnectedLayer(Neurons1,'Name','CriticActionFC1')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

% Create critic representation
criticOptions = rlRepresentationOptions('Optimizer','adam','LearnRate',1e-2, ... 
                                        'GradientThreshold',1,'L2RegularizationFactor',2e-4,'UseDevice',"gpu");
critic = rlQValueRepresentation(criticNetwork,obsInfo,...
                           actInfo,'Observation',{'observation'}, ...
                          'Action',{'action'},criticOptions);

agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN',false, ...    
    'TargetUpdateMethod',"smoothing", ...
    'TargetSmoothFactor',1e-3, ...   
    'ExperienceBufferLength',1e5, ...
    'DiscountFactor',0.99, ...
    'MiniBatchSize',256*4,...
    'SaveExperienceBufferWithAgent',true,...
    'ResetExperienceBufferBeforeTraining',true,...
    'NumStepsToLookAhead',10);
agentOpts.EpsilonGreedyExploration.Epsilon = .9;
agentOpts.EpsilonGreedyExploration.EpsilonDecay = 1e-5;

agent = rlDQNAgent(critic,agentOpts);


%% training the agent
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 100000, ...
    'MaxStepsPerEpisode', 150, ...
    'Verbose', false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','GlobalStepCount',...
    'StopTrainingValue',2e5,...
    'SaveAgentCriteria','GlobalStepCount',...
    'SaveAgentValue',2e5,...
    'SaveAgentDirectory','C:\Users\jande\OneDrive\Documenten\Master\UCSC\ObstacleAvoidance\HybridDQN\AgentsOA2021',...
    'ScoreAveragingWindowLength',100); 

% plot(env) % Uncomment to visualize
doTraining = true;
if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    load('AgentsOA2021/...') % Choose an agent to start training from
    saved_agent.AgentOptions.EpsilonGreedyExploration.Epsilon = 0.7;  
    saved_agent.AgentOptions.EpsilonGreedyExploration.EpsilonDecay = 1e-5;
    trainingStats = train(saved_agent,env,trainOpts);
end


%% Simulate DQN agent 
rng(211)
simOptions = rlSimulationOptions('MaxSteps',150);
plot(env)
experience = sim(env,agent,simOptions);

totalReward = sum(experience.Reward)