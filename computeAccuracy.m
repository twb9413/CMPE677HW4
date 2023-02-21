function accuracy = computeAccuracy(Xdata,theta, y)
% computeAccuracy: given Xdata, theta and 
    class_pred = sigmoid(Xdata * theta) >= 0.5;
    correct = sum(class_pred == y);
    accuracy = correct / length(y);
end

