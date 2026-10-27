function h = digestMD5(b)
% DIGESTMD5  Stand-in for MathWorks' undocumented digestMD5 (P-code in Spinach):
% the 16-byte MD5 digest (uint8) of a byte vector.
    hx = hash('md5', char(b(:).'));
    h = uint8(sscanf(reshape(hx, 2, []), '%2x')).';
end
