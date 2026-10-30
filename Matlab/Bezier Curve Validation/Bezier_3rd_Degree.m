% Creating empty arrays for storing control point input
array_x = [];
array_y = [];

% Getting user input for the x and y coordinates of each control point
for i = 1:4
    value_x = input(sprintf('Enter x coordinate of %dth control point: ', i));
    array_x = [array_x, value_x];
    value_y = input(sprintf('Enter y coordinate of %dth control point: ', i));
    array_y = [array_y, value_y];
end
% Creating empty arrays for storing x and y coordinates of the curve points
x = [];
y = [];

% Creating a variable u and initializing it to zero
u = 0;

% while loop to draw a Cubic curve using Bezier curve
while u < 1
   % Cubic Curve
   put_x = (1 - u)^3 * array_x(1) + 3 * u * (1 - u)^2 * array_x(2) + 3 * u^2 * (1 - u) * array_x(3) + u^3 * array_x(4);
   put_y = (1 - u)^3 * array_y(1) + 3 * u * (1 - u)^2 * array_y(2) + 3 * u^2 * (1 - u) * array_y(3) + u^3 * array_y(4);


    
    % Appending x and y coordinates of the curve points
    x = [x, put_x];
    y = [y, put_y];
    
    % Incrementing u by 0.01
    u = u + 0.001;
end
disp("Array X:");
disp(x);
disp("Array Y:");
disp(y);
% Plotting the cubic Bezier curve
plot(array_x, array_y,'*','DisplayName', 'Control Points');
title('Third Degree Bezier Curve');
xlabel('XCoordinate');
ylabel('Ycoordinate');
hold on;
plot(array_x, array_y, 'b', 'DisplayName', 'Control Ploygon');
plot(x, y, 'g', 'DisplayName', 'Cubic Bezier Curve','linewidth', 1);
legend;
hold off;
