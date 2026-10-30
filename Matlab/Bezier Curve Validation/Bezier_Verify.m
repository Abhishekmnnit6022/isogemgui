% Creating empty arrays for storing control point input
array_x = [];
array_y = [];

% Getting user input for the x and y coordinates of each control point
for i = 1:2
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

% Creating a while loop to draw a straight line using Bezier curve
while u < 1
    % Computing x and y coordinates of the curve points using Bezier curve formula
    put_x = (1-u) * array_x(1) + u * array_x(2);
    put_y = (1 - u) * array_y(1) + u * array_y(2);
    
    % Appending x and y coordinates of the curve points to their respective arrays
    x = [x, put_x];
    y = [y, put_y];
    
    % Incrementing u by 0.01
    u = u + 0.001;
end
disp(x);
disp(y);

% Plotting the Bezier curve
plot(array_x, array_y, '*');
title('First Degree Bezier Curve');
xlabel('XCoordinate');
ylabel('Ycoordinate');
hold on;
plot(array_x, array_y, 'b', 'DisplayName', 'Control Point');
plot(x, y, 'g', 'DisplayName', 'Curve','linewidth', 1);
legend;
hold off;
