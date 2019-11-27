function a = apply(n,param)
%TRIBAS.APPLY Apply transfer function to inputs

% Copyright 2012-2015 The MathWorks, Inc.

  a = log(1+exp(n));
  a(isnan(n)) = nan;
end
