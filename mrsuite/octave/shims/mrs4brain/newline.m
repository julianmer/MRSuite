function c = newline()
% NEWLINE  MRS4Brain-scoped stand-in for MATLAB's newline.
% MATLAB's newline is char(10). MRS4Brain only uses it as "literal" + newline and
% in strcmp(...,{'#','$',newline}); in MATLAB the "literal" is a string, so the
% sum is string concatenation. Octave reads "literal" as char, which would make
% the sum numeric; returning the newline as a string stand-in (mrsuite_text)
% restores the concatenation. strcmp in this scope reads it as char(10).
    c = mrsuite_text(char(10));
end
