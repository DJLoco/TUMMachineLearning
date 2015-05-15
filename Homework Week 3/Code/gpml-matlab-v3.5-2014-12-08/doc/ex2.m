disp(' '); disp('clear all, close all')
clear all, close all
write_fig = 0;
disp(' ')

disp('n1 = 80; n2 = 40;                   % number of data points from each class')
n1 = 80; n2 = 40;
disp('S1 = eye(2); S2 = [1 0.95; 0.95 1];           % the two covariance matrices')
S1 = eye(2); S2 = [1 0.95; 0.95 1];
disp('m1 = [0.75; 0]; m2 = [-0.75; 0];                            % the two means')
m1 = [0.75; 0]; m2 = [-0.75; 0];
disp(' ')

%%%%%%% Generating of data with respect to covariance matrices S1, S2 %%%%%%
disp('x1 = bsxfun(@plus, chol(S1)''*gpml_randn(0.2, 2, n1), m1);')
x1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, n1), m1);
disp('x2 = bsxfun(@plus, chol(S2)''*gpml_randn(0.3, 2, n2), m2);')         
x2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, n2), m2);         
disp(' ')

%%%%%%% Displaying input data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('x = [x1 x2]''; y = [-ones(1,n1) ones(1,n2)]'';')
x = [x1 x2]'; y = [-ones(1,n1) ones(1,n2)]';
figure(6)
disp('plot(x1(1,:), x1(2,:), ''b+''); hold on;');
plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on
disp('plot(x2(1,:), x2(2,:), ''r+'');');
plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12);
disp(' ')

%%%%%%% Generating test inputs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('[t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);')
[t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);
disp('t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs')
t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs
disp('tmm = bsxfun(@minus, t, m1'');')
tmm = bsxfun(@minus, t, m1');
disp('p1 = n1*exp(-sum(tmm*inv(S1).*tmm/2,2))/sqrt(det(S1));')
p1 = n1*exp(-sum(tmm*inv(S1).*tmm/2,2))/sqrt(det(S1));
disp('tmm = bsxfun(@minus, t, m2'');')
tmm = bsxfun(@minus, t, m2');
disp('p2 = n2*exp(-sum(tmm*inv(S2).*tmm/2,2))/sqrt(det(S2));')
p2 = n2*exp(-sum(tmm*inv(S2).*tmm/2,2))/sqrt(det(S2));
set(gca, 'FontSize', 24)
disp('contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.1:0.1:0.9])')
contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.1:0.1:0.9])
[c h] = contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.5 0.5]);
set(h, 'LineWidth', 2)
colorbar
grid
axis([-4 4 -4 4])
if write_fig, print -depsc f6.eps; end
disp(' '); disp('Hit any key to continue...'); pause


%%%%%%% Initializing parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
infs = {@infEP @infLaplace};
liks = {{@likErf []} {@likLogistic []} {@likUni []} {@likGauss log(0.1)}};
covs = {{@covSEard log([1 1 1])} {@covSEiso log([0.9; 2])}};


%%%%% Different inference function %%%%%%%%%%%%%%%%%%%%%%%%%%%%
l = 1;
k = 1;
for i = 1:length(infs)
    infFunc = infs{i}
    likFunc = liks{l}{1}
    hyp.lik = liks{l}{2}
    covFunc = covs{k}{1}
    hyp.cov = covs{k}{2}
    disp(' ')
    disp('meanfunc = @meanConst; hyp.mean = 0;')
    meanfunc = @meanConst; hyp.mean = 0;
    
    hyp = minimize(hyp, @gp, -40, infFunc, meanfunc, covFunc, likFunc, x, y);
    [a b c d lp] = gp(hyp, infFunc, meanfunc, covFunc, likFunc, x, y, t, ones(n,1));
    disp(' ')
    figure()
    set(gca, 'FontSize', 24)
    plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on
    plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12)
    contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
    [c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
    set(h, 'LineWidth', 2)
    colorbar
    grid
    axis([-4 4 -4 4])
    if write_fig, print -depsc f7.eps; end
    disp(' '); disp('Hit any key to continue...'); pause
end

%%%%% Different likelihood function %%%%%%%%%%%%%%%%%%%%%%%%%%%%
i = 1;
k = 1;
for l = 1:length(liks)
    infFunc = infs{i}
    likFunc = liks{l}{1}
    hyp.lik = liks{l}{2}
    covFunc = covs{k}{1}
    hyp.cov = covs{k}{2}
    disp(' ')
    disp('meanfunc = @meanConst; hyp.mean = 0;')
    meanfunc = @meanConst; hyp.mean = 0;
    
    hyp = minimize(hyp, @gp, -40, infFunc, meanfunc, covFunc, likFunc, x, y);
    [a b c d lp] = gp(hyp, infFunc, meanfunc, covFunc, likFunc, x, y, t, ones(n,1));
    disp(' ')
    figure()
    set(gca, 'FontSize', 24)
    plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on
    plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12)
    contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
    [c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
    set(h, 'LineWidth', 2)
    colorbar
    grid
    axis([-4 4 -4 4])
    if write_fig, print -depsc f7.eps; end
    disp(' '); disp('Hit any key to continue...'); pause
end

%%%%% Different kernel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i = 1;
l = 1;
for k = 1:length(covs)
    infFunc = infs{i}
    likFunc = liks{l}{1}
    hyp.lik = liks{l}{2}
    covFunc = covs{k}{1}
    hyp.cov = covs{k}{2}
    disp(' ')
    disp('meanfunc = @meanConst; hyp.mean = 0;')
    meanfunc = @meanConst; hyp.mean = 0;
    
    hyp = minimize(hyp, @gp, -40, infFunc, meanfunc, covFunc, likFunc, x, y);
    [a b c d lp] = gp(hyp, infFunc, meanfunc, covFunc, likFunc, x, y, t, ones(n,1));
    disp(' ')
    figure()
    set(gca, 'FontSize', 24)
    plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on
    plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12)
    contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
    [c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
    set(h, 'LineWidth', 2)
    colorbar
    grid
    axis([-4 4 -4 4])
    if write_fig, print -depsc f7.eps; end
    disp(' '); disp('Hit any key to continue...'); pause
end