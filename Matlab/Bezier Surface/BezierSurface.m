clear all;

uPTS = 3;
wPTS = 3;
uCELLS = 120;
wCELLS = 80;

% Rectangle Surface
% x = [0 2 3; 0 2 3; 0 2 3];
% y = [0 0 0; 1 1 1; 2 2 2];
% z = zeros(3, 3);

% Pycharm same input
x = [-0.5 -2 0; 1, 1, 1; 2, 2, 2];
y = [2 1 0; 2 8 -1; 2 1 1];
z = [1 -1 2; 0 -8.5 2; 0.5 1 2];


% Using Random control points
% x = rand(uPTS, wPTS);
% y = rand(uPTS, wPTS);
% z = rand(uPTS, wPTS);

%other varibales
n = uPTS -1;
m = wPTS -1;
u = linspace(0,1, uCELLS);
w = linspace(0,1, wCELLS);
i = 0:n;
j = 0:m;

% Binomial coefficients
cu = factorial(n)./(factorial(i).*factorial(n-i));
cw = factorial(m)./(factorial(j).*factorial(m-j));

%Bernstein Basis Polynomials
for k = 1:uPTS
    J(k,:) = cu(k) * u.^i(k).*(1-u).^(n - i(k));
end

for k= 1:wPTS
    K(k,:) = cu(k)*w.^j(k).*(1-w).^(m - j(k));
end

% Empty array for bezier curve
xBezier = zeros(uCELLS, wCELLS);
yBezier = zeros(uCELLS, wCELLS);
zBezier = zeros(uCELLS, wCELLS);

% Main loop
for i = 1:n+1
    
    for j = 1:m+1
        xBezier = J(i,:)'* K(j,:) * x(i, j) + xBezier;
        yBezier = J(i,:)'* K(j,:) * y(i, j) + yBezier;
        zBezier = J(i,:)'* K(j,:) * z(i, j) + zBezier;
    end
end

%plotting Basis function first
figure(1); plot(u,J)
figure(2); plot(w,K)

xlabel('XCoordinate');
ylabel('Ycoordinate');
%plotting control points
figure(3); mesh(xBezier,yBezier,zBezier)
    hold on;
    plot3(x,y,z,'o', 'MarkerFaceColor', 'k')
    hold off;

