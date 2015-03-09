function [result] = d_tanh(x)
    result = 1 - tanh(x).^2;
end