function p = mrsuite_serial_registry(op, p)
% Holds the current serial pool (gcp returns it, parpool sets it, delete clears it).
    persistent current
    switch op
        case 'get', p = current;
        case 'set', current = p;
        case 'clear', current = [];
    end
end
