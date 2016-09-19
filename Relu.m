classdef Relu
    properties
        name;
        input;
        output;
        delta;
    end
    
    methods
        function layer = Relu(name)
            layer.name = name;
        end
        
        function layer = forward(layer, input)
            % Your codes here
            layer.output = max(zeros(size(input, 1), size(input, 2), 'single'), input);
        end
        
        function layer = backprop(layer, delta)
            % Your codes here
            tem = layer.output;
            tem(find(tem<=0)) = 0;
            tem(find(tem>0)) = 1;
            layer.delta = delta .* tem;
        end
    end
end

            
            