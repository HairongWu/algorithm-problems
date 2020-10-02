# Object and type object interface
`Py_ssize_t` is a signed integral type such that `sizeof(Py_ssize_t) == sizeof(size_t)`.  C99 doesn't define such a thing directly (size_t is an unsigned integral type).  See PEP 353 for details.
```c
#ifdef MS_WIN64
typedef __int64 ssize_t;
#else
typedef _W64 int ssize_t;
#endif

#define HAVE_SSIZE_T 1

#ifdef HAVE_SSIZE_T
typedef ssize_t         Py_ssize_t;
#elif SIZEOF_VOID_P == SIZEOF_SIZE_T
typedef Py_intptr_t     Py_ssize_t;
#else
#   error "Python needs a typedef for Py_ssize_t in pyport.h."
#endif
```
