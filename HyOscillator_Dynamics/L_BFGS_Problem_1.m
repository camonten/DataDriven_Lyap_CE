%% L-BFGS (Algorithm 5.4)
clear all; clc;
set(groot,'defaultAxesTickLabelInterpreter','latex');
% termination tolerance
tol = 1e-10;
% maximum number of allowed iterations
maxiter = 1000;
% initial condition
n = 1000;
alpha = 10;
x0 = rand(n, 1);
xopt = ones(n ,1);
% initialize gradient norm, optimization vector, iteration counter
gnorm = inf; x = x0; niter = 0;
% define objective function
f = @(x) sum(alpha * (x(2:2:end) - x(1:2:end-1).^2).^2 + (1 - x(1:2:end-1)).^2);
grad = @(x) grad_f(x, alpha);





function gradient = grad_f(x, p)
    n = length(x);
    gradient = zeros(size(x));
    
    % Compute gradient for each pair of elements (x_{2i-1}, x_{2i})
    for i = 1:n/2
        % Compute partial derivatives
        df_dx2i = 2 * p * (x(2*i) - x(2*i - 1)^2);  % Partial derivative wrt x_{2i}
        df_dx2i_1 = -4 * p * x(2*i - 1) * (x(2*i) - x(2*i - 1)^2) - 2 * (1 - x(2*i - 1));  % Partial derivative wrt x_{2i-1}
        
        % Update gradient
        gradient(2*i) = df_dx2i;
        gradient(2*i - 1) = df_dx2i_1;
    end
end


function r = recursion_two_loop(gradient, S, Y, m)
    q = gradient;
    a = zeros(1, m);
    rou = 1./dot(Y, S, 2);
    for i = 1:m
        a(m - i + 1) = rou(m - i + 1) * dot(S(m - i + 1, :), q);
        q = q - a(m - i + 1) * Y(m - i + 1, :);
    end
    
    H_k0 = dot(S(m, :), Y(m, :))/dot(Y(m, :), Y(m, :));
    r = H_k0 * q;
    
    for i = 1:m
        beta = rou(i) * dot(Y(i, :), r);
        r = r + (a(i) - beta) * S(i, :);
    end
end

function [xopt, xstore] = LBFGS(x, xopt, f, grad, gnorm, tol, m)
    p = -grad(x);
    niter = 0;

    Y = {};
    S = {};
    while gnorm >= tol

        % calculate gradient:
        g = grad(x);
        gnorm = norm(g);
        % calculate step size
        alpha = StepLength(f, grad, p, x);
        % take step:
        x_plus = x + alpha * p;

        % append to S
        S(end + 1) = alpha * p;
        % calculate new gradient
        g_plus = grad(x_plus);
        % append to Y
        Y(end + 1) = g_plus - g;
        
        % discard s_{k-m} and y_{k-m} from S and Y
        if niter > m
            p = -recursion_two_loop(g, S, Y, m);

            % discard
            S(niter - m) = [];
            Y(niter - m) = [];
        else
            p = -recursion_two_loop(g, S, Y, niter);
        end

        % update variables for next iteration
%         p = -recursion_two_loop(g, S, Y, m);
        x = x_plus;
        niter = niter + 1;
    end
end




