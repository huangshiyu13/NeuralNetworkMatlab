classdef euclideanLossLayer
    
    properties
        name;
        input;
        input_shape;
        
        output;
        delta;
        loss;
        accuracy;
    end
    
    methods
        function layer = euclideanLossLayer(name)
            layer.name = name;
        end
    
        function layer = forward(layer, input)
            % Your codes here
            %[max_val, layer.output] = max(input);
            layer.input = input;
            layer.output = input;
        end
        
        function layer = backprop(layer, label)
            % Your codes here
            lab = zeros(size(layer.output));
            for i = 1 : 1 : size(label, 2)
               lab(label(i), i) = 1; 
            end
            layer.loss = 1/2 * ones(1, 10) * (lab - layer.output).^2;
            [~, check] = max(layer.output);
            layer.accuracy = sum(check == label) / size(label, 2);
            layer.delta = -(lab - layer.output);
        end
    end
end