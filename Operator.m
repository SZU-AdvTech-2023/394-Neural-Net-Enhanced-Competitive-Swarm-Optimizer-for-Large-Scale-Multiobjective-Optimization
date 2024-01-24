function Offspring = Operator(Loser,Winner)

    %% Parameter setting
    LoserDec  = Loser.decs;
    WinnerDec = Winner.decs;
    [N,D]     = size(LoserDec);
	LoserVel  = Loser.adds(zeros(N,D));
    WinnerVel = Winner.adds(zeros(N,D));
    Problem   = PROBLEM.Current();
    ptrain = LoserDec;
    ttrain = WinnerDec;
    ttest = WinnerDec;
    T_sim = BPNN(ptrain,ttest);
    T_sim = T_sim';

    %% Competitive swarm optimizer
    r1     = repmat(rand(N,1),1,D);
    r2     = repmat(rand(N,1),1,D);
    OffVel = r1.*LoserVel + r2.*(WinnerDec-LoserDec);
    OffDec = LoserDec + OffVel + r1.*(OffVel-LoserVel);
    
    %% Add the winners
    OffDec = [OffDec;T_sim];
    OffVel = [OffVel;WinnerVel];
    %% Polynomial mutation
    Lower  = repmat(Problem.lower,2*N,1);
    Upper  = repmat(Problem.upper,2*N,1);
    disM   = 20;
    Site   = rand(2*N,D) < 1/D;
    mu     = rand(2*N,D);
    temp   = Site & mu<=0.5;
    OffDec       = max(min(OffDec,Upper),Lower);
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                   (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp  = Site & mu>0.5; 
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                   (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
	Offspring = SOLUTION(OffDec,OffVel);
end