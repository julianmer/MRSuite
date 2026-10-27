function b = serializeToBytes(A)
% SERIALIZETOBYTES  Stand-in for MathWorks' undocumented serializeToBytes (shipped
% to Spinach as P-code, which Octave cannot run): a byte stream (uint8 row) that
% identifies the value, here Octave's binary save format of A. Same value -> same
% bytes (Octave's binary format has no timestamp); the bytes differ from
% MATLAB's, so hashes (cache keys) differ from MATLAB's too.
    f = [tempname() '.bin'];
    save('-binary', f, 'A');
    fid = fopen(f, 'r'); b = fread(fid, Inf, 'uint8=>uint8').'; fclose(fid); delete(f);
end
