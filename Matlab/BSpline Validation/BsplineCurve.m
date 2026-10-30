function BsplineCurve()
    knots = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5];
    degree = 2;
    control_points = [5, 5, 50, 50, 70, 80, 90, 100; 20, 70, 2, 40, 40, 2, 30, 30];

    basis = plot_bspline_basis(knots, degree);
    plot_bspline_curve(basis, control_points);
end

function basis = bspline_basis(i, p, t, knots)
    basis = zeros(size(t));

    for j = 1:numel(t)
        basis(j) = bspline_recursive(i, p, t(j), knots);
    end
end

function result = bspline_recursive(i, p, t, knots)
    if p == 0
        if knots(i) <= t && t < knots(i + 1)
            result = 1;
        else
            result = 0;
        end
    else
        denom1 = knots(i + p) - knots(i);
        denom2 = knots(i + p + 1) - knots(i + 1);

        if denom1 == 0
            coeff1 = 0;
        else
            coeff1 = (t - knots(i)) / denom1;
        end
        
        if denom2 == 0
            coeff2 = 0;
        else
            coeff2 = (knots(i + p + 1) - t) / denom2;
        end

        result = coeff1 * bspline_recursive(i, p - 1, t, knots) + coeff2 * bspline_recursive(i + 1, p - 1, t, knots);
    end
end

function basis = plot_bspline_basis(knots, degree)
    t = linspace(knots(1), knots(end), 1000);
    basis = zeros(length(knots) - degree - 1, length(t));

    for i = 1:length(knots) - degree - 1
        basis(i, :) = bspline_basis(i, degree, t, knots);
    end

    figure;
    for i = 1:length(knots) - degree - 1
        plot(t, basis(i, :), 'DisplayName', sprintf('N_%d_%d', i, degree));
        hold on;
    end
    hold off;
    
    legend('show');
    xlabel('t');
    ylabel('Basis value');
    title(sprintf('B-Spline Basis Functions of degree %d with knot Vector %s', degree, mat2str(knots)));
    
    disp('Basis Matrix:');
    disp(basis);
end

function plot_bspline_curve(basis, control_points)
    t = linspace(0, 1, 1000);
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
    hold off;
    
    legend('show');
    xlabel('x');
    ylabel('y');
    title(sprintf('B-Spline Curve having Degree %d for knots %s', degree, mat2str(knots)));
    
    disp('B-Spline Curve Point Value of x, y:');
    disp(curve_points);
end

BsplineCurve();
