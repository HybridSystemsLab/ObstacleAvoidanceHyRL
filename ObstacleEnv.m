classdef ObstacleEnv < rl.env.MATLABEnvironment
     
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties    
        MaxForce = 1
        h = [];
        Rewards = [];
        Observ = [];
    end
    
    properties
        % Initialize system state [x,y]'
        State = zeros(2,1)
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = ObstacleEnv()
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([3 1]);
            ObservationInfo.Name = 'Car Observations';
            
            % Initialize Action settings   
            ActionInfo = rlFiniteSetSpec(linspace(-1,1,5));
            
            ActionInfo.Name = 'Control Action';
           
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % Initialize property values and pre-compute necessary values
            updateActionInfo(this);
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            LoggedSignals = [];
            dt = 0.05;
            delta = 0.75;    
        
            % Unpack state vector
            X = this.State(1);
            Y = this.State(2);
            
            
            % Apply motion equations   
            d_ob = sqrt((X-1.5)^2+(Y)^2)-delta;
            if d_ob < 0
                    d_ob = 0;
            end
            d_go = sqrt((3-X)^2+Y^2);
         
            Observation = [d_ob; d_go; Y];
            
            this.State(1) = X + 1*dt;
            this.State(2) = Y + Action*dt;
            this.Observ = Observation;
            
            % Get reward
            Reward = getReward(this);
            this.Rewards = Reward + this.Rewards;
            % Check terminal condition
            X = this.State(1);
            Y = this.State(2);
            if d_ob == 0 ||  abs(Y) >= 1.5 || X >= 3 
                IsDone = true;
            else
                IsDone = false;
            end
            % Use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            delta = 0.75;
            % X
            X0 = 0.5*rand(); 
            ystart = 1-rand()*2;
            Y0 = ystart(randi(length(ystart)));
            this.Rewards = 0;
            
            d_ob0 = sqrt((X0-1.5)^2+(Y0)^2)-delta;
            if d_ob0 < 0
                d_ob0 = 0;
            end
            d_go0 = sqrt((3-X0)^2+Y0^2);
            
            % Return initial environment state variables as logged signals.
            InitialObservation = [d_ob0; d_go0; Y0];
            this.State = [X0; Y0];
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods               
        % Helper methods to create the environment
        function force = getForce(this,action)
            if action(1) < -1 || action(1) > 1 %|| action(2) < -1 || action(2) > 1
               error('Action must be %g for going left and %g for going right.',...
                    -this.MaxForce,this.MaxForce);
            end
            force = action;          
        end
        % update the action info based on max force
        function updateActionInfo(this)
        end

        function Reward = getReward(this)
        delta = 0.75;
        
        d_ob = this.Observ(1);
        d_go = this.Observ(2);

        % Get reward
        
        Barrier = (d_ob-2*delta)^2-log(d_ob);
        Reward = -d_go-0.1*Barrier+3.5;
        if Reward < 0
            Reward = 0;
        end
            
        end
        
        % (optional) Visualization method
        function plot(this)
%             % Initiate the visualization, uncomment this to visualize
%             this.h = figure;
%             
%             % Update the visualization
%             envUpdatedCallback(this)
        end
        
 
    end
    
    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
            
        % Set the visualization figure as the current figure, uncomment this
        % to visualize
%         figure(this.h);
%         clf
% 
%         % Extract the cart position and pole angle
%         X = this.State(1);
%         Y = this.State(2);
% 
%         
%         Z1 = linspace(0,3.5,100);
%         plot(X,Y,'o','MarkerSize',7,'MarkerFaceColor','blue')
%         grid on
%         hold on
%         pgon1 = polyshape([3 3 3.1 3.1],[0.1,-.1 -.1 .1]);
%         pos = [0.75 -0.75 1.5 1.5]; 
%         rectangle('Position',pos,'Curvature',[1 1],'FaceColor','red','edgecolor','none')
%         plot(pgon1)
%         plot(Z1,ones(100,1)*1.5,'--','Color','r')
%         plot(Z1,ones(100,1)*-1.5,'--','Color','r')
%         ylim([-1.55 1.55])
% 
%         hold off
            
            
        end
    end
end
