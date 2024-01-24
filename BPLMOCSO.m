classdef BPLMOCSO < ALGORITHM
    methods
        function main(Algorithm,Problem)
            %% Generate random population
            [V,Problem.N] = UniformPoint(Problem.N,Problem.M);
            Population    = Problem.Initialization();

            %% Optimization
            while Algorithm.NotTerminated(Population)
                Fitness = calFitness(Population.objs);
                if length(Population) >= 2
                    Rank = randperm(length(Population),floor(length(Population)/2)*2);
                else
                    Rank = [1,1];
                end
                Loser  = Rank(1:end/2);
                Winner = Rank(end/2+1:end);
                Change = Fitness(Loser) >= Fitness(Winner);
                Temp   = Winner(Change);
                Winner(Change) = Loser(Change);
                Loser(Change)  = Temp;
                Offspring      = Operator(Population(Loser),Population(Winner));
                Population     = EnvironmentalSelection([Population,Offspring],V,(Problem.FE/Problem.maxFE)^2);
            end
        end
    end
end

function Fitness = calFitness(PopObj)
% Calculate the fitness by shift-based density

    N      = size(PopObj,1);
    fmax   = max(PopObj,[],1);
    fmin   = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
    Dis    = inf(N);
    for i = 1 : N
        SPopObj = max(PopObj,repmat(PopObj(i,:),N,1));
        for j = [1:i-1,i+1:N]
            Dis(i,j) = norm(PopObj(i,:)-SPopObj(j,:));
        end
    end
    Fitness = min(Dis,[],2);
end