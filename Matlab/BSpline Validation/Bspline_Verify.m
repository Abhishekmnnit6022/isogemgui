knots = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 4.995];
degree = 2;
control_points = [5, 5, 50, 50, 70, 80, 90, 100;
                  20,70, 2, 40, 40, 2, 30, 30];

plot_bspline_basis(knots, degree);
plot_bspline_curve(basis, control_points);

function basis_value = bspline_basis(i, p, t, knots)
    if p == 0
        if knots(i) <= t && t < knots(i + 1)
            basis_value = 1;
        else
            basis_value = 0;
        end
    else
        denom1 = knots(i + p) - knots(i);
        denom2 = knots(i + p + 1) - knots(i + 1);
        
        if denom1 == 0 && denom2 == 0
            coeff1 = 0;
            coeff2 = 0;
        elseif denom1 == 0
            coeff1 = 0;
            coeff2 = (knots(i + p + 1) - t) / denom2;
        elseif denom2 == 0
            coeff1 = (t - knots(i)) / denom1;
            coeff2 = 0;
        else
            coeff1 = (t - knots(i)) / denom1;
            coeff2 = (knots(i + p + 1) - t) / denom2;
        end
        
        basis_value = coeff1 * bspline_basis(i, p - 1, t, knots) + coeff2 * bspline_basis(i + 1, p - 1, t, knots);
    end
end

function basis = plot_bspline_basis(knots, degree)
    t = linspace(knots(1), knots(end), 1000);
    basis = zeros(length(knots) - degree - 1, length(t));
    
    for i = 1:length(knots) - degree - 1
        for j = 1:length(t)
            basis(i, j) = bspline_basis(i, degree, t(j), knots);
        end
    end
    
    figure;
    for i = 1:length(knots) - degree - 1
        plot(t, basis(i, :), 'DisplayName', ['N_' num2str(i) '_' num2str(degree)]);
        hold on;
    end
    
    legend;
    xlabel('t');
    ylabel('Basis value');
    title(['B-Spline Basis Functions of degree ' num2str(degree) ' with knot Vector ' mat2str(knots)]);
    hold off;
    
    disp('Basis Matrix:');
    disp(basis);
end

function plot_bspline_curve(basis, control_points)
    t = linspace(0, 1, 999);
    curve_points = zeros(size(control_points, 1), length(t));
    
    for i = 1:length(t)
        for j = 1:size(control_points, 2)
            curve_points(:, i) = curve_points(:, i) + basis(j, i) * control_points(:, j);
        end
    end
    
    figure;
    plot(control_points(1, :), control_points(2, :), 'bo-', 'DisplayName', 'Control Points');
    hold on;
    plot(curve_points(1, :), curve_points(2, :), 'r-', 'DisplayName', 'B-Spline Curve');
    legend;
    xlabel('x');
    ylabel('y');
    title(['B-Spline Curve with Control Points']);
    hold off;
    
    disp('B-Spline Curve Point Value of x,y:');
    disp(curve_points);
end


