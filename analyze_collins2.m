function results = analyze_collins2(data)

% Analyze Collins (2018) data.

if nargin < 1
    data = load_data('collins18');
end

beta = linspace(0.1,15,50);

% for set size = 3
% < 0.2 complexity 13 (block 4 and block 10)
% > 0.5 complexity 29(block 14), 31 (block 1, 10, 12), 51 

% for set size = 6
% < 0.2 complexity 13, 26
% > 0.8 complexity 60, 77
for s = 51
    B = unique(data(s).learningblock);
    cond = zeros(length(B),1);
    R_data =zeros(length(B),1);
    V_data =zeros(length(B),1);
    for b = 1:length(B)
        ix = data(s).learningblock==B(b) & data(s).phase==0;
        state = data(s).state(ix);
        c = data(s).corchoice(ix);
        action = data(s).action(ix);
        action(action==-1) = 2;
        for a = 1:max(action)
            Pa(a) = mean(action==a); % marginal action probability
            %PaR(a,:,b) = movmean(action==a,5); % running action probability
        end
        R_data(b) = mutual_information(state,action,0.1);
        V_data(b) = mean(data(s).reward(ix));
        
        S = unique(state);
        Q = zeros(length(S),3);
        Ps = zeros(1,length(S));
        for i = 1:length(S)
            ii = state==S(i);
            Ps(i) = mean(ii);
            a = c(ii); a = a(1);
            Q(i,a) = 1;
            
            for a = 1:max(action)
                Pas(i,a) = mean(action(state==i)==a); % conditioned on state
            end
        end
          
        KL = nansum(Pas.*log(Pas./Pa),2); % KL divergence between policies for each state
        C(b) = sum(Ps'.*KL);
        
        figure; hold on;
        P  = [Pa; Pas];
        for p = 1:size(P,1)
            subplot(size(P,1),1,p); hold on; prettyplot;
            bar(P(p,:));
            if p >1 ylabel(strcat('P(a|s_',num2str(p-1),')')); title(num2str(KL(p-1,:))); else ylabel('P(a)'); title(num2str(mean(KL))); end
            
        end
        equalabscissa(size(P,1),1)
        suptitle(num2str(R_data(b)))
        
        [R(b,:),V(b,:)] = blahut_arimoto(Ps,Q,beta);
        
        if length(S)==3
            cond(b) = 1;
        else
            cond(b) = 2;
        end
        
        clear Pa Pas
    end
    
    figure; plot(C',R_data,'ko'); prettyplot; axis square; dline;
    xlabel('empirically computed I(S;A)')
    ylabel('matlab function estimated I(S;A)')
    
    R_data = C';
    
    for c = 1:2
        results.R(s,:,c) = nanmean(R(cond==c,:));
        results.V(s,:,c) = nanmean(V(cond==c,:));
        results.R_data(s,c) = nanmean(R_data(cond==c));
        results.V_data(s,c) = nanmean(V_data(cond==c));
   
    end
    
    clear R V
    
end

p = signrank(results.R_data(:,1),results.R_data(:,2))

R = squeeze(nanmean(results.R));
V = squeeze(nanmean(results.V));
for c = 1:2
    Vd2(:,c) =  interp1(R(:,c),V(:,c),results.R_data(:,c));
    results.bias(:,c) = results.V_data(:,c) - Vd2(:,c);
end

[r,p] = corr([results.V_data(:,1); results.V_data(:,2)],[Vd2(:,1); Vd2(:,2)])
[r,p] = corr([results.R_data(:,1); results.R_data(:,2)],abs([results.bias(:,1); results.bias(:,2)]))