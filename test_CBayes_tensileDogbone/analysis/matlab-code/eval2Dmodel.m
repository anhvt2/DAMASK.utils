function varargout = eval2Dmodel(p,model,varargin)

N = size(p,1);
if nargin == 3
    noise = varargin{1};
else 
    noise = zeros(N,1);
end
M = size(noise,2);

switch model
    case 'rosenbrock'
        a = 1;
        b = 100;
        qvals = (a-p(:,1)).^2 + b*(p(:,2)-p(:,1).^2).^2;
    case 'peaks'
        p1 = gaussian(2.0, p(:,1), p(:,2), 0.25, 0.75, 0.15,0.15);
        p2 = gaussian(3.0, p(:,1), p(:,2), 0.75, 0.75, 0.2,0.2);
        p3 = gaussian(2.5, p(:,1), p(:,2), 0.33, 0.33, 0.1,0.1);
        p4 = gaussian(-1.0, p(:,1), p(:,2), 0.8, 0.4, 0.1,0.2);

        qvals = p1+p2+p3+p4;
    case 'linear1'
        
        A = [ 1 2 ; -1 1];
        qvals = A*p';
        qvals = qvals(1,:)';
        
    case 'poly'
        
%         qvals = (1*p(:,1) - 2*p(:,1).^2 - 3*p(:,1).^3).*(2*p(:,2) + p(:,2).^2 - 2*p(:,2).^4);
        qvals = (1*p(:,1) + sin(3*pi*p(:,1))).*(2*p(:,2) + p(:,2).^2 - 2*p(:,2).^4);
        
    case 'linear'
        
        qvals = 0.0*p(:,1) - 2.0*p(:,2);
        
    case 'hump'
        
        qvals = tanh(40*(p(:,1)-1/2));
        
    case 'nlinv'
        qvals = zeros(N*M,1);
        dvals = zeros(N*M,2);
        pp = 0;
        for k = 1:N
            for j = 1:M
                x = [1;1];
                data = [1;1];
                
                resid = data - [p(k,1)*x(1)^2 + x(2)^2;x(1)^2 - p(k,2)*x(2)^2];
                J = [2*p(k,1)*x(1) 2*x(2);2*x(1) -2*p(k,2)*x(2)];
                
                iter = 0;
                while norm(resid) > 1e-10 && iter < 20
                    
                    iter = iter + 1;
                    dx = J\resid;
                    x = x+dx;
                    resid = data - [p(k,1)*x(1)^2 + x(2)^2;x(1)^2 - p(k,2)*x(2)^2];
                    J = [2*p(k,1)*x(1) 2*x(2);2*x(1) -2*p(k,2)*x(2)];
                    
                end
                
                qvals(pp+1,1) = x(2)+noise(k,j);
%                 qvals(pp+1,2) = x(2);
                pp = pp+1;
                
            end
        end
        
    case 'discont'
        x1 = 0+1*p(:,1);
        x2 = 0+1*p(:,2);
        
        f1 = exp(-x1.^2-x2.^2) - x1.^3-x2.^3;
        f2 = 1+f1+1/8*x2.^2;
        f = zeros(N,1);
        
        for j = 1:N
            if ((3*x1(j)+2*x2(j))>=0) && ((-x1(j)+0.3*x2(j))<0)
                f(j) = f1(j)-2;
            elseif  ((3*x1(j)+2*x2(j))>=0) && ((-x1(j)+0.3*x2(j))>=0)
                f(j) = 2*f2(j);
            elseif (x1(j)+1)^2+(x2(j)+1)^2 <= 0.95^2
                f(j) = 2*f1(j)+4;
            else
                f(j) = f1(j);
            end
        end
        
        qvals = f;
end

if nargout == 1
    varargout{1} = qvals;
    return;    
elseif nargout == 2
    varargout{1} = qvals;
    varargout{2} = dvals;
    return;    
end

