function [u, Q, Qm]= Findu_2_2021(params,Ksi)
%% Extracting the parameters of the neural net
% Enable this is CPU is used for training
% % State path:
% w_s1 = squeeze(params{1,1}); 
% b_s1 = squeeze(params{1,2}); 
% 
% w_s2 = squeeze(params{1,3});
% b_s2 = squeeze(params{1,4});
% 
% % Action path:
% w_a1 = squeeze(params{1,5});
% b_a1 = squeeze(params{1,6});
% 
% % Common path:
% w_c1 = squeeze(params{1,7});
% b_c1 = squeeze(params{1,8});
% 
% w_c2 = squeeze(params{1,9});
% b_c2 = squeeze(params{1,10});

% Enable this is GPU is used for training
% State path: 
w_s1 = gather(squeeze(params{1,1})); 
b_s1 = gather(squeeze(params{1,2})); 

w_s2 = gather(squeeze(params{1,3}));
b_s2 = gather(squeeze(params{1,4}));

% Action path:
w_a1 = gather(squeeze(params{1,5}));
b_a1 = gather(squeeze(params{1,6}));

% Common path:
w_c1 = gather(squeeze(params{1,7}));
b_c1 = gather(squeeze(params{1,8}));

w_c2 = gather(squeeze(params{1,9}));
b_c2 = gather(squeeze(params{1,10}));

%% States

Q = zeros(5,1);
Actions = linspace(-1,1,5);

for ii = 1:length(Q)
    h_s1 = leakyrelu(w_s1.'*Ksi+b_s1,0.001);
    h_s2 = (w_s2.'*h_s1+b_s2);

    h_a1 = ((Actions(ii)*w_a1)+b_a1);

    h_add = leakyrelu(h_s2 + h_a1,0.001);

    h_c1 = leakyrelu(w_c1.'*h_add + b_c1,0.001);
    Q(ii) = w_c2.'*h_c1 + b_c2;
end
Qm = max(Q);
u = Actions(find(Q == max(Q)));
if length(u)>1
    u = u(randi(length(u)));
end

end
function H = leakyrelu(CC,leak)
H = zeros(size(CC));
[length, width] = size(CC);
for II = 1:length
    for ii = 1:width
        if CC(II,ii) < 0
            H(II,ii) = leak*CC(II,ii);
        else
            H(II,ii) = CC(II,ii);
        end
    end
end
end