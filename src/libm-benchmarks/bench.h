// ARRAY SIZE: N * veclen
#define N 10000
// NUM ITERATIONS
#define ITER 1000
// TOTAL NUMBER OF ITERATIONS
#define NITER (N * ITER)

#define callFuncSLEEF1_1(funcName, name, xmin, xmax, ulp, arg, type_in, type_out) ({	\
      __attribute__((unused)) type_out out = funcName(*((type_in *) arg));	\
      printf("%s\n", #funcName);						\
      uint64_t t0 = Sleef_currentTimeMicros();					\
      for(int j=0;j<ITER;j++) {							\
	type_in *p = (type_in *)(arg);						\
	for(int i=0;i<N;i++){							\
		out = funcName(*p++);						\
	}									\
      }										\
      uint64_t t1 =  Sleef_currentTimeMicros();					\
      uint64_t dt = t1-t0;							\
      fprintf(fp, name ", %.3g, %.3g, %gulps, %g\n",				\
	      (double)xmin, (double)xmax, ulp, (double) dt / NITER); \
    })

#define callFuncSLEEF1_2(funcName, name, xmin, xmax, ymin, ymax, ulp, arg1, arg2, type_in, type_out) ({ \
      __attribute__((unused)) type_out out = funcName(*((type_in *) arg1), *((type_in *) arg2));	\
      printf("%s\n", #funcName);									\
      uint64_t t0 = Sleef_currentTimeMicros();								\
      for(int j=0;j<ITER;j++) {										\
	type_in *p1 = (type_in *)(arg1), *p2 = (type_in *)(arg2);					\
	for(int i=0;i<N;i++){										\
		out = funcName(*p1++, *p2++);								\
	}												\
      }													\
      uint64_t t1 =  Sleef_currentTimeMicros();								\
      uint64_t dt = t1-t0;										\
      fprintf(fp, name ", %.3g, %.3g, %.3g, %.3g, %gulps, %g\n",					\
	      (double)xmin, (double)xmax, (double)ymin, (double)ymax, ulp, (double) dt / NITER);	\
    })

#define callFuncSVML1_1(funcName, name, xmin, xmax, arg, type_in) ({	\
      printf("%s\n", #funcName);					\
      uint64_t t = Sleef_currentTimeMicros();				\
      for(int j=0;j<ITER;j++) {					\
	type_in *p = (type_in *)(arg);					\
	for(int i=0;i<N;i++) funcName(*p++);			\
      }									\
      fprintf(fp, name ", %.3g, %.3g, %gulps, %g\n",			\
	      (double)xmin, (double)xmax, (double)SVMLULP, (double)(Sleef_currentTimeMicros() - t) / NITER); \
    })

#define callFuncSVML2_1(funcName, name, xmin, xmax, arg, type_in) ({	\
      printf("%s\n", #funcName);					\
      uint64_t t = Sleef_currentTimeMicros();				\
      for(int j=0;j<ITER;j++) {					\
	type_in *p = (type_in *)(arg), c;					\
	for(int i=0;i<N;i++) funcName(&c, *p++);			\
      }									\
      fprintf(fp, name ", %.3g, %.3g, %gulps, %g\n",			\
	      (double)xmin, (double)xmax, (double)SVMLULP, (double)(Sleef_currentTimeMicros() - t) / NITER); \
    })

#define callFuncSVML1_2(funcName, name, xmin, xmax, ymin, ymax, arg1, arg2, type_in) ({ \
      printf("%s\n", #funcName);					\
      uint64_t t = Sleef_currentTimeMicros();				\
      for(int j=0;j<ITER;j++) {					\
	type_in *p1 = (type_in *)(arg1), *p2 = (type_in *)(arg2);		\
	for(int i=0;i<N;i++) funcName(*p1++, *p2++);		\
      }									\
      fprintf(fp, name ", %.3g, %.3g, %.3g, %.3g, %gulps, %g\n",	\
	      (double)xmin, (double)xmax, (double)ymin, (double)ymax, (double)SVMLULP, (double)(Sleef_currentTimeMicros() - t) / NITER); \
    })
