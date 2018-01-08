# Orthonormal Tangent Basis:

This is just a simple example of a vectorized basis construction that accepts a tangent vector as well as a normal. It can be useful when trying to evaluate anisotropic reflection models that needs to correlate with dPdu or dPdv. Recently there were two really great papers published (http://jcgt.org/published/0006/01/01/ and http://jcgt.org/published/0006/01/02/) on the subject that I'd recommend reading. However these algorithms don't support the use of tangent vectors, and in such cases something like this might be a reasonable alternative.

## Required Dependencies:

* CMake 2.8
* SSE 4.1