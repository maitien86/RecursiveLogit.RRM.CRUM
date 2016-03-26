%   Get attribute 
function Att = getAtt()

    global incidenceFull;
    global EstimatedTime;
    global TurnAngles;
    global LeftTurn;
    global Uturn;
    global LSatt;
    global isLinkSizeInclusive;
    [lastIndexNetworkState, nsize] = size(incidenceFull);
    mu = 1;
    Incidence = 1/mu * incidenceFull;
    Att(1) = Matrix2D(Incidence .* EstimatedTime);
    Att(2) = Matrix2D(Incidence .* TurnAngles);%% left turn
    Att(3) = Matrix2D(Incidence .* Uturn);
    Att(4) = Matrix2D(Incidence .* LeftTurn);%% link constant
    if isLinkSizeInclusive == 1
        Att(5) = Matrix2D(Incidence .* LSatt);
    end
end
