package org.naokishibata.examples;

import static org.naokishibata.sleef.FastMath.*;

/** A class to perform correctness and speed tests for FastMath class
 *
 * @author Naoki Shibata
 */
public class FastMathTest {
    static boolean isnan(double d) { return d != d; }

    static boolean cmpDenorm(double x, double y) {
	if (isnan(x) && isnan(y)) return true;
	if (x == Double.POSITIVE_INFINITY && y == Double.POSITIVE_INFINITY) return true;
	if (x == Double.NEGATIVE_INFINITY && y == Double.NEGATIVE_INFINITY) return true;
	if (!isnan(x) && !isnan(y) && !Double.isInfinite(x) && !Double.isInfinite(y)) return true;
	return false;
    }

    /** Perform correctness and speed tests. The accuracy is checked
     * by comparing results with the standard math library. Note that
     * calculation by the standard math library also has error, and
     * the reported error is basically 1 + the calculation error by
     * the FastMath methods.
     */
    public static void main(String[] args) throws Exception {
	System.out.println();

	//

	System.out.println("Denormal test atan2(y, x)");
	System.out.println();

	System.out.print("If y is +0 and x is -0, +pi is returned ... ");
	System.out.println((atan2(+0.0, -0.0) == Math.PI) ? "OK" : "NG");
	//System.out.println(atan2(+0.0, -0.0));

	System.out.print("If y is -0 and x is -0, -pi is returned ... ");
	System.out.println((atan2(-0.0, -0.0) == -Math.PI) ? "OK" : "NG");
	//System.out.println(atan2(-0.0, -0.0));

	System.out.print("If y is +0 and x is +0, +0 is returned ... ");
	System.out.println(isPlusZero(atan2(+0.0, +0.0)) ? "OK" : "NG");
	//System.out.println(atan2(+0.0, +0.0));

	System.out.print("If y is -0 and x is +0, -0 is returned ... ");
	System.out.println(isMinusZero(atan2(-0.0, +0.0)) ? "OK" : "NG");
	//System.out.println(atan2(-0.0, +0.0));

	System.out.print("If y is positive infinity and x is negative infinity, +3*pi/4 is returned ... ");
	System.out.println((atan2(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY) == 3*Math.PI/4) ? "OK" : "NG");
	//System.out.println(atan2(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY));

	System.out.print("If y is negative infinity and x is negative infinity, -3*pi/4 is returned ... ");
	System.out.println((atan2(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY) == -3*Math.PI/4) ? "OK" : "NG");
	//System.out.println(atan2(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY));

	System.out.print("If y is positive infinity and x is positive infinity, +pi/4 is returned ... ");
	System.out.println((atan2(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY) == Math.PI/4) ? "OK" : "NG");
	//System.out.println(atan2(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY));

	System.out.print("If y is negative infinity and x is positive infinity, -pi/4 is returned ... ");
	System.out.println((atan2(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY) == -Math.PI/4) ? "OK" : "NG");
	//System.out.println(atan2(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY));

	{
	    System.out.print("If y is +0 and x is less than 0, +pi is returned ... ");

	    double[] ya = { +0.0 };
	    double[] xa = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (atan2(ya[j], xa[i]) != Math.PI) {
			System.out.print("[atan2(" + ya[j] + ", " + xa[i] + ") = " + atan2(ya[j], xa[i]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If y is -0 and x is less than 0, -pi is returned ... ");


	    double[] ya = { -0.0 };
	    double[] xa = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (atan2(ya[j], xa[i]) != -Math.PI) {
			System.out.print("[atan2(" + ya[j] + ", " + xa[i] + ") = " + atan2(ya[j], xa[i]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If y is less than 0 and x is 0, -pi/2 is returned ... ");

	    double[] ya = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };
	    double[] xa = { +0.0, -0.0 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (atan2(ya[j], xa[i]) != -Math.PI/2) {
			System.out.print("[atan2(" + ya[j] + ", " + xa[i] + ") = " + atan2(ya[j], xa[i]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If y is greater than 0 and x is 0, pi/2 is returned ... ");


	    double[] ya = { 100000.5, 100000, 3, 2.5, 2, 1.5, 1.0, 0.5 };
	    double[] xa = { +0.0, -0.0 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (atan2(ya[j], xa[i]) != Math.PI/2) {
			System.out.print("[atan2(" + ya[j] + ", " + xa[i] + ") = " + atan2(ya[j], xa[i]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If y is greater than 0 and x is -0, pi/2 is returned ... ");

	    double[] ya = { 100000.5, 100000, 3, 2.5, 2, 1.5, 1.0, 0.5 };
	    double[] xa = { -0.0 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (atan2(ya[j], xa[i]) != Math.PI/2) {
			System.out.print("[atan2(" + ya[j] + ", " + xa[i] + ") = " + atan2(ya[j], xa[i]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If y is positive infinity, and x is finite, pi/2 is returned ... ");

	    double[] ya = { Double.POSITIVE_INFINITY };
	    double[] xa = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (atan2(ya[j], xa[i]) != Math.PI/2) {
			System.out.print("[atan2(" + ya[j] + ", " + xa[i] + ") = " + atan2(ya[j], xa[i]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If y is negative infinity, and x is finite, -pi/2 is returned ... ");

	    double[] ya = { Double.NEGATIVE_INFINITY };
	    double[] xa = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (atan2(ya[j], xa[i]) != -Math.PI/2) {
			System.out.print("[atan2(" + ya[j] + ", " + xa[i] + ") = " + atan2(ya[j], xa[i]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If y is a finite value greater than 0, and x is negative infinity, +pi is returned ... ");

	    double[] ya = { 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
	    double[] xa = { Double.NEGATIVE_INFINITY };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (atan2(ya[j], xa[i]) != Math.PI) {
			System.out.print("[atan2(" + ya[j] + ", " + xa[i] + ") = " + atan2(ya[j], xa[i]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If y is a finite value less than 0, and x is negative infinity, -pi is returned ... ");

	    double[] ya = { -0.5, -1.5, -2.0, -2.5, -3.0, -100000, -100000.5 };
	    double[] xa = { Double.NEGATIVE_INFINITY };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (atan2(ya[j], xa[i]) != -Math.PI) {
			System.out.print("[atan2(" + ya[j] + ", " + xa[i] + ") = " + atan2(ya[j], xa[i]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If y is a finite value greater than 0, and x is positive infinity, +0 is returned ... ");

	    double[] ya = { 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
	    double[] xa = { Double.POSITIVE_INFINITY };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!isPlusZero(atan2(ya[j], xa[i]))) {
			System.out.print("[atan2(" + ya[j] + ", " + xa[i] + ") = " + atan2(ya[j], xa[i]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If y is a finite value less than 0, and x is positive infinity, -0 is returned ... ");

	    double[] ya = { -0.5, -1.5, -2.0, -2.5, -3.0, -100000, -100000.5 };
	    double[] xa = { Double.POSITIVE_INFINITY };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!isMinusZero(atan2(ya[j], xa[i]))) {
			System.out.print("[atan2(" + ya[j] + ", " + xa[i] + ") = " + atan2(ya[j], xa[i]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If x is NaN, a NaN is returned ... ");

	    double[] ya = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5, Double.NaN };
	    double[] xa = { Double.NaN };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!Double.isNaN(atan2(ya[j], xa[i]))) {
			System.out.print("[atan2(" + ya[j] + ", " + xa[i] + ") = " + atan2(ya[j], xa[i]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If y is a NaN, the result is a NaN ... ");

	    double[] ya = { Double.NaN };
	    double[] xa = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5, Double.NaN };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!Double.isNaN(atan2(ya[j], xa[i]))) {
			System.out.print("[atan2(" + ya[j] + ", " + xa[i] + ") = " + atan2(ya[j], xa[i]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	System.out.println();
	System.out.println("end of atan2 denormal test");
	System.out.println();

	//

	System.out.println("Denormal test pow(x, y)");
	System.out.println();

	//System.out.print("If the result overflows, a range error occurs, and the functions return HUGE_VAL with the mathematically correct sign ... ");
	//System.out.print("If result underflows, and is not representable, a range error occurs, and 0.0 is returned ... ");

	System.out.print("If x is +1 and y is a NaN, the result is 1.0 ... ");
	System.out.println(pow(1, Double.NaN) == 1.0 ? "OK" : "NG");

	System.out.print("If y is 0 and x is a NaN, the result is 1.0 ... ");
	System.out.println(pow(Double.NaN, 0) == 1.0 ? "OK" : "NG");

	System.out.print("If x is -1, and y is positive infinity, the result is 1.0 ... ");
	System.out.println(pow(-1, Double.POSITIVE_INFINITY) == 1.0 ? "OK" : "NG");

	System.out.print("If x is -1, and y is negative infinity, the result is 1.0 ... ");
	System.out.println(pow(-1, Double.NEGATIVE_INFINITY) == 1.0 ? "OK" : "NG");

	{
	    System.out.print("If x is a finite value less than 0, and y is a finite non-integer, a NaN is returned ... ");

	    double[] xa = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };
	    double[] ya = { -100000.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 100000.5 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!Double.isNaN(pow(xa[i], ya[j]))) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If x is a NaN, the result is a NaN ... ");

	    double[] xa = { Double.NaN };
	    double[] ya = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!Double.isNaN(pow(xa[i], ya[j]))) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If y is a NaN, the result is a NaN ... ");

	    double[] xa = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5, -0.0, +0.0, 0.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
	    double[] ya = { Double.NaN };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!Double.isNaN(pow(xa[i], ya[j]))) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If x is +0, and y is an odd integer greater than 0, the result is +0 ... ");

	    double[] xa = { +0.0 };
	    double[] ya = { 1, 3, 5, 7, 100001 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!isPlusZero(pow(xa[i], ya[j]))) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If x is -0, and y is an odd integer greater than 0, the result is -0 ... ");

	    double[] xa = { -0.0 };
	    double[] ya = { 1, 3, 5, 7, 100001 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!isMinusZero(pow(xa[i], ya[j]))) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If x is 0, and y greater than 0 and not an odd integer, the result is +0 ... ");

	    double[] xa = { +0.0, -0.0 };
	    double[] ya = { 0.5, 1.5, 2.0, 2.5, 4.0, 100000, 100000.5 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!isPlusZero(pow(xa[i], ya[j]))) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If the absolute value of x is less than 1, and y is negative infinity, the result is positive infinity ... ");

	    double[] xa = { -0.999, -0.5, -0.0, +0.0, +0.5, +0.999 };
	    double[] ya = { Double.NEGATIVE_INFINITY };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (pow(xa[i], ya[j]) != Double.POSITIVE_INFINITY) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If the absolute value of x is greater than 1, and y is negative infinity, the result is +0 ... ");

	    double[] xa = { -100000.5, -100000, -3, -2.5, -2, -1.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
	    double[] ya = { Double.NEGATIVE_INFINITY };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!isPlusZero(pow(xa[i], ya[j]))) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If the absolute value of x is less than 1, and y is positive infinity, the result is +0 ... ");

	    double[] xa = { -0.999, -0.5, -0.0, +0.0, +0.5, +0.999 };
	    double[] ya = { Double.POSITIVE_INFINITY };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!isPlusZero(pow(xa[i], ya[j]))) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If the absolute value of x is greater than 1, and y is positive infinity, the result is positive infinity ... ");

	    double[] xa = { -100000.5, -100000, -3, -2.5, -2, -1.5, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };
	    double[] ya = { Double.POSITIVE_INFINITY };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (pow(xa[i], ya[j]) != Double.POSITIVE_INFINITY) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If x is negative infinity, and y is an odd integer less than 0, the result is -0 ... ");

	    double[] xa = { Double.NEGATIVE_INFINITY };
	    double[] ya = { -100001, -5, -3, -1 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!isMinusZero(pow(xa[i], ya[j]))) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If x is negative infinity, and y less than 0 and not an odd integer, the result is +0 ... ");

	    double[] xa = { Double.NEGATIVE_INFINITY };
	    double[] ya = { -100000.5, -100000, -4, -2.5, -2, -1.5, -0.5 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!isPlusZero(pow(xa[i], ya[j]))) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If x is negative infinity, and y is an odd integer greater than 0, the result is negative infinity ... ");

	    double[] xa = { Double.NEGATIVE_INFINITY };
	    double[] ya = { 1, 3, 5, 7, 100001 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (pow(xa[i], ya[j]) != Double.NEGATIVE_INFINITY) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If x is negative infinity, and y greater than 0 and not an odd integer, the result is positive infinity ... ");

	    double[] xa = { Double.NEGATIVE_INFINITY };
	    double[] ya = { 0.5, 1.5, 2, 2.5, 3.5, 4, 100000, 100000.5 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (pow(xa[i], ya[j]) != Double.POSITIVE_INFINITY) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If x is positive infinity, and y less than 0, the result is +0 ... ");

	    double[] xa = { Double.POSITIVE_INFINITY };
	    double[] ya = { -100000.5, -100000, -3, -2.5, -2, -1.5, -1.0, -0.5 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!isPlusZero(pow(xa[i], ya[j]))) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If x is positive infinity, and y greater than 0, the result is positive infinity ... ");

	    double[] xa = { Double.POSITIVE_INFINITY };
	    double[] ya = { 0.5, 1, 1.5, 2.0, 2.5, 3.0, 100000, 100000.5 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (pow(xa[i], ya[j]) != Double.POSITIVE_INFINITY) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If x is +0, and y is an odd integer less than 0, +HUGE_VAL is returned ... ");

	    double[] xa = { +0.0 };
	    double[] ya = { -100001, -5, -3, -1 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (pow(xa[i], ya[j]) != Double.POSITIVE_INFINITY) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If x is -0, and y is an odd integer less than 0, -HUGE_VAL is returned ... ");

	    double[] xa = { -0.0 };
	    double[] ya = { -100001, -5, -3, -1 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (pow(xa[i], ya[j]) != Double.NEGATIVE_INFINITY) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If x is 0, and y is less than 0 and not an odd integer, +HUGE_VAL is returned ... ");

	    double[] xa = { +0.0, -0.0 };
	    double[] ya = { -100000.5, -100000, -4, -2.5, -2, -1.5, -0.5 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (pow(xa[i], ya[j]) != Double.POSITIVE_INFINITY) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("If the result overflows, the functions return HUGE_VAL with the mathematically correct sign ... ");

	    double[] xa = { 1000, -1000 };
	    double[] ya = { 1000, 1000.5, 1001 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		for(int j=0;j<ya.length && success;j++) {
		    if (!cmpDenorm(pow(xa[i], ya[j]), Math.pow(xa[i], ya[j]))) {
			System.out.print("[x = " + xa[i] + ", y = " + ya[j] + ", pow(x,y) = " + pow(xa[i], ya[j]) + "] ");
			success = false;
			break;
		    }
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	System.out.println();
	System.out.println("End of pow denormal test");
	System.out.println();
	
	//

	{
	    System.out.print("sin denormal test ... ");

	    double[] xa = { Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		if (!cmpDenorm(sin(xa[i]), Math.sin(xa[i]))) {
		    System.out.print("[x = " + xa[i] + ", func(x) = " + sin(xa[i]) + "] ");
		    success = false;
		    break;
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("sin in sincos denormal test ... ");

	    double[] xa = { Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		double2 q = sincos(xa[i]);
		if (!cmpDenorm(q.x, Math.sin(xa[i]))) {
		    System.out.print("[x = " + xa[i] + ", func(x) = " + q.x + "] ");
		    success = false;
		    break;
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("cos denormal test ... ");

	    double[] xa = { Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		if (!cmpDenorm(cos(xa[i]), Math.cos(xa[i]))) {
		    System.out.print("[x = " + xa[i] + ", func(x) = " + cos(xa[i]) + "] ");
		    success = false;
		    break;
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("cos in sincos denormal test ... ");

	    double[] xa = { Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		double2 q = sincos(xa[i]);
		if (!cmpDenorm(q.y, Math.cos(xa[i]))) {
		    System.out.print("[x = " + xa[i] + ", func(x) = " + q.y + "] ");
		    success = false;
		    break;
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("tan denormal test ... ");

	    double[] xa = { Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Math.PI/2, -Math.PI/2 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		if (!cmpDenorm(tan(xa[i]), Math.tan(xa[i]))) {
		    System.out.print("[x = " + xa[i] + ", func(x) = " + tan(xa[i]) + "] ");
		    success = false;
		    break;
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("asin denormal test ... ");

	    double[] xa = { Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 2, -2 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		if (!cmpDenorm(asin(xa[i]), Math.asin(xa[i]))) {
		    System.out.print("[x = " + xa[i] + ", func(x) = " + asin(xa[i]) + "] ");
		    success = false;
		    break;
		}
		//System.out.print("[x = " + xa[i] + ", func(x) = " + asin(xa[i]) + ", correct = " + Math.asin(xa[i]) + "] ");
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("acos denormal test ... ");

	    double[] xa = { Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 2, -2 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		if (!cmpDenorm(acos(xa[i]), Math.acos(xa[i]))) {
		    System.out.print("[x = " + xa[i] + ", func(x) = " + acos(xa[i]) + "] ");
		    success = false;
		    break;
		}
		//System.out.print("[x = " + xa[i] + ", func(x) = " + acos(xa[i]) + ", correct = " + Math.acos(xa[i]) + "] ");
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("atan denormal test ... ");

	    double[] xa = { Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		if (!cmpDenorm(atan(xa[i]), Math.atan(xa[i]))) {
		    System.out.print("[x = " + xa[i] + ", func(x) = " + atan(xa[i]) + "] ");
		    success = false;
		    break;
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("log denormal test ... ");

	    double[] xa = { Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 0, -1 };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		if (!cmpDenorm(log(xa[i]), Math.log(xa[i]))) {
		    System.out.print("[x = " + xa[i] + ", func(x) = " + log(xa[i]) + "] ");
		    success = false;
		    break;
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	{
	    System.out.print("exp denormal test ... ");

	    double[] xa = { Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY };

	    boolean success = true;
	    for(int i=0;i<xa.length && success;i++) {
		if (!cmpDenorm(exp(xa[i]), Math.exp(xa[i]))) {
		    System.out.print("[x = " + xa[i] + ", func(x) = " + exp(xa[i]) + "] ");
		    success = false;
		    break;
		}
	    }

	    System.out.println(success ? "OK" : "NG");
	}

	//

	System.out.println();
	System.out.println("Accuracy test (max error in ulp)");
	System.out.println();

	double max;

	max = 0;

	for(double d = -10;d < 10;d += 0.000001) {
	    double q = sin(d);
	    double c = Math.sin(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	for(double d = -10000;d < 10000;d += 0.001) {
	    double q = sin(d);
	    double c = Math.sin(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("sin : " + max);

	max = 0;

	for(double d = -10;d < 10;d += 0.000001) {
	    double q = cos(d);
	    double c = Math.cos(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	for(double d = -10000;d < 10000;d += 0.001) {
	    double q = cos(d);
	    double c = Math.cos(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("cos : " + max);

	max = 0;

	for(double d = -10;d < 10;d += 0.000001) {
	    double2 q = sincos(d);
	    double c = Math.sin(d);
	    double u = Math.abs((q.x - c) / Math.ulp(c));
	    max = max(max, u);
	}

	for(double d = -10000;d < 10000;d += 0.001) {
	    double2 q = sincos(d);
	    double c = Math.sin(d);
	    double u = Math.abs((q.x - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("sin in sincos : " + max);

	max = 0;

	for(double d = -10;d < 10;d += 0.000001) {
	    double2 q = sincos(d);
	    double c = Math.cos(d);
	    double u = Math.abs((q.y - c) / Math.ulp(c));
	    max = max(max, u);
	}

	for(double d = -10000;d < 10000;d += 0.001) {
	    double2 q = sincos(d);
	    double c = Math.cos(d);
	    double u = Math.abs((q.y - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("cos in sincos : " + max);

	max = 0;

	for(double d = -10;d < 10;d += 0.000001) {
	    double q = tan(d);
	    double c = Math.tan(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	for(double d = -10000;d < 10000;d += 0.001) {
	    double q = tan(d);
	    double c = Math.tan(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("tan : " + max);

	max = 0;

	for(double d = -1;d < 1;d += 0.0000001) {
	    double q = asin(d);
	    double c = Math.asin(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("asin : " + max);

	max = 0;

	for(double d = -1;d < 1;d += 0.0000001) {
	    double q = acos(d);
	    double c = Math.acos(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("acos : " + max);

	max = 0;

	for(double d = -10;d < 10;d += 0.000001) {
	    double q = atan(d);
	    double c = Math.atan(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	for(double d = -10000;d < 10000;d += 0.001) {
	    double q = atan(d);
	    double c = Math.atan(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("atan : " + max);

	max = 0;

	for(double d = 0.001;d < 10;d += 0.000001) {
	    double q = log(d);
	    double c = Math.log(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	for(double d = 0.001;d < 100000;d += 0.01) {
	    double q = log(d);
	    double c = Math.log(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("log : " + max);

	max = 0;

	for(double d = -10;d < 10;d += 0.000001) {
	    double q = exp(d);
	    double c = Math.exp(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	for(double d = -700;d < 700;d += 0.0001) {
	    double q = exp(d);
	    double c = Math.exp(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("exp : " + max);

	//

	max = 0;

	for(double y = -10;y < 10;y += 0.01) {
	    for(double x = -10;x < 10;x += 0.01) {
		double q = atan2(y, x);
		double c = Math.atan2(y, x);
		double u = Math.abs((q - c) / Math.ulp(c));
		max = max(max, u);
	    }
	}

	for(double y = -1000;y < 1000;y += 1.01) {
	    for(double x = -1000;x < 1000;x += 1.01) {
		double q = atan2(y, x);
		double c = Math.atan2(y, x);
		double u = Math.abs((q - c) / Math.ulp(c));
		max = max(max, u);
	    }
	}

	System.out.println("atan2 : " + max);

	max = 0;

	for(double y = 0;y < 100;y += 0.05) {
	    for(double x = -100;x < 100;x += 0.05) {
		double q = pow(x, y);
		double c = Math.pow(x, y);
		double u = Math.abs((q - c) / Math.ulp(c));
		max = max(max, u);
	    }
	}

	System.out.println("pow : " + max);

	//

	max = 0;

	for(double d = -700;d < 700;d += 0.00001) {
	    double q = sinh(d);
	    double c = Math.sinh(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("sinh : " + max);

	//

	max = 0;

	for(double d = -700;d < 700;d += 0.00001) {
	    double q = cosh(d);
	    double c = Math.cosh(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("cosh : " + max);

	//

	max = 0;

	for(double d = -700;d < 700;d += 0.00001) {
	    double q = tanh(d);
	    double c = Math.tanh(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("tanh : " + max);

	//

	max = 0;

	for(double d = 0;d < 10000;d += 0.001) {
	    double q = sqrt(d);
	    double c = Math.sqrt(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("sqrt : " + max);

	//

	max = 0;

	for(double d = -10000;d < 10000;d += 0.001) {
	    double q = cbrt(d);
	    double c = Math.cbrt(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("cbrt : " + max);

	/*
	
	max = 0;

	for(double d = -700;d < 700;d += 0.0002) {
	    double q = asinh(d);
	    double c = Math.asinh(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("asinh : " + max);

	//

	max = 0;

	for(double d = 1;d < 700;d += 0.0001) {
	    double q = acosh(d);
	    double c = Math.acosh(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("acosh : " + max);

	//

	max = 0;

	for(double d = -700;d < 700;d += 0.0002) {
	    double q = atanh(d);
	    double c = Math.atanh(d);
	    double u = Math.abs((q - c) / Math.ulp(c));
	    max = max(max, u);
	}

	System.out.println("atanh : " + max);

	*/

	System.out.println();
	System.out.println("Speed test");
	System.out.println();

	double total = 0;

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += Math.sin(d + 0);
		sum += Math.sin(d + 1);
		sum += Math.sin(d + 2);
		sum += Math.sin(d + 3);
		sum += Math.sin(d + 4);
		sum += Math.sin(d + 5);
		sum += Math.sin(d + 6);
		sum += Math.sin(d + 7);
		sum += Math.sin(d + 8);
		sum += Math.sin(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("sin standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += sin(d + 0);
		sum += sin(d + 1);
		sum += sin(d + 2);
		sum += sin(d + 3);
		sum += sin(d + 4);
		sum += sin(d + 5);
		sum += sin(d + 6);
		sum += sin(d + 7);
		sum += sin(d + 8);
		sum += sin(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("sin sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += Math.cos(d + 0);
		sum += Math.cos(d + 1);
		sum += Math.cos(d + 2);
		sum += Math.cos(d + 3);
		sum += Math.cos(d + 4);
		sum += Math.cos(d + 5);
		sum += Math.cos(d + 6);
		sum += Math.cos(d + 7);
		sum += Math.cos(d + 8);
		sum += Math.cos(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("cos standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += cos(d + 0);
		sum += cos(d + 1);
		sum += cos(d + 2);
		sum += cos(d + 3);
		sum += cos(d + 4);
		sum += cos(d + 5);
		sum += cos(d + 6);
		sum += cos(d + 7);
		sum += cos(d + 8);
		sum += cos(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("cos sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += Math.tan(d + 0);
		sum += Math.tan(d + 1);
		sum += Math.tan(d + 2);
		sum += Math.tan(d + 3);
		sum += Math.tan(d + 4);
		sum += Math.tan(d + 5);
		sum += Math.tan(d + 6);
		sum += Math.tan(d + 7);
		sum += Math.tan(d + 8);
		sum += Math.tan(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("tan standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += tan(d + 0);
		sum += tan(d + 1);
		sum += tan(d + 2);
		sum += tan(d + 3);
		sum += tan(d + 4);
		sum += tan(d + 5);
		sum += tan(d + 6);
		sum += tan(d + 7);
		sum += tan(d + 8);
		sum += tan(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("tan sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += Math.sin(d + 0); sum += Math.cos(d + 0);
		sum += Math.sin(d + 1); sum += Math.cos(d + 1);
		sum += Math.sin(d + 2); sum += Math.cos(d + 2);
		sum += Math.sin(d + 3); sum += Math.cos(d + 3);
		sum += Math.sin(d + 4); sum += Math.cos(d + 4);
		sum += Math.sin(d + 5); sum += Math.cos(d + 5);
		sum += Math.sin(d + 6); sum += Math.cos(d + 6);
		sum += Math.sin(d + 7); sum += Math.cos(d + 7);
		sum += Math.sin(d + 8); sum += Math.cos(d + 8);
		sum += Math.sin(d + 9); sum += Math.cos(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("sin + cos, standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		double2 r;

		r = sincos(d + 0); sum += r.x + r.y;
		r = sincos(d + 1); sum += r.x + r.y;
		r = sincos(d + 2); sum += r.x + r.y;
		r = sincos(d + 3); sum += r.x + r.y;
		r = sincos(d + 4); sum += r.x + r.y;
		r = sincos(d + 5); sum += r.x + r.y;
		r = sincos(d + 6); sum += r.x + r.y;
		r = sincos(d + 7); sum += r.x + r.y;
		r = sincos(d + 8); sum += r.x + r.y;
		r = sincos(d + 9); sum += r.x + r.y;
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("sincos sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -1;d < 0;d += 0.0000005) {
		sum += Math.asin(d + 0.0);
		sum += Math.asin(d + 0.1);
		sum += Math.asin(d + 0.2);
		sum += Math.asin(d + 0.3);
		sum += Math.asin(d + 0.4);
		sum += Math.asin(d + 0.5);
		sum += Math.asin(d + 0.6);
		sum += Math.asin(d + 0.7);
		sum += Math.asin(d + 0.8);
		sum += Math.asin(d + 0.9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("asin standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -1;d < 0;d += 0.0000005) {
		sum += asin(d + 0.0);
		sum += asin(d + 0.1);
		sum += asin(d + 0.2);
		sum += asin(d + 0.3);
		sum += asin(d + 0.4);
		sum += asin(d + 0.5);
		sum += asin(d + 0.6);
		sum += asin(d + 0.7);
		sum += asin(d + 0.8);
		sum += asin(d + 0.9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("asin sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -1;d < 0;d += 0.0000005) {
		sum += Math.acos(d + 0.0);
		sum += Math.acos(d + 0.1);
		sum += Math.acos(d + 0.2);
		sum += Math.acos(d + 0.3);
		sum += Math.acos(d + 0.4);
		sum += Math.acos(d + 0.5);
		sum += Math.acos(d + 0.6);
		sum += Math.acos(d + 0.7);
		sum += Math.acos(d + 0.8);
		sum += Math.acos(d + 0.9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("acos standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -1;d < 0;d += 0.0000005) {
		sum += acos(d + 0.0);
		sum += acos(d + 0.1);
		sum += acos(d + 0.2);
		sum += acos(d + 0.3);
		sum += acos(d + 0.4);
		sum += acos(d + 0.5);
		sum += acos(d + 0.6);
		sum += acos(d + 0.7);
		sum += acos(d + 0.8);
		sum += acos(d + 0.9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("acos sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += Math.atan(d + 0);
		sum += Math.atan(d + 1);
		sum += Math.atan(d + 2);
		sum += Math.atan(d + 3);
		sum += Math.atan(d + 4);
		sum += Math.atan(d + 5);
		sum += Math.atan(d + 6);
		sum += Math.atan(d + 7);
		sum += Math.atan(d + 8);
		sum += Math.atan(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("atan standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += atan(d + 0);
		sum += atan(d + 1);
		sum += atan(d + 2);
		sum += atan(d + 3);
		sum += atan(d + 4);
		sum += atan(d + 5);
		sum += atan(d + 6);
		sum += atan(d + 7);
		sum += atan(d + 8);
		sum += atan(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("atan sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double y = -10;y < 10;y += 0.01) {
		for(double x = -10;x < 10;x += 0.01) {
		    sum += Math.atan2(y + 0, x);
		    sum += Math.atan2(y + 1, x);
		    sum += Math.atan2(y + 2, x);
		    sum += Math.atan2(y + 3, x);
		    sum += Math.atan2(y + 4, x);
		    sum += Math.atan2(y + 5, x);
		    sum += Math.atan2(y + 6, x);
		    sum += Math.atan2(y + 7, x);
		    sum += Math.atan2(y + 8, x);
		    sum += Math.atan2(y + 9, x);
		}
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("atan2 standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double y = -10;y < 10;y += 0.01) {
		for(double x = -10;x < 10;x += 0.01) {
		    sum += atan2(y + 0, x);
		    sum += atan2(y + 1, x);
		    sum += atan2(y + 2, x);
		    sum += atan2(y + 3, x);
		    sum += atan2(y + 4, x);
		    sum += atan2(y + 5, x);
		    sum += atan2(y + 6, x);
		    sum += atan2(y + 7, x);
		    sum += atan2(y + 8, x);
		    sum += atan2(y + 9, x);
		}
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("atan2 sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = 0.001;d < 10;d += 0.000001) {
		sum += Math.log(d + 0);
		sum += Math.log(d + 1);
		sum += Math.log(d + 2);
		sum += Math.log(d + 3);
		sum += Math.log(d + 4);
		sum += Math.log(d + 5);
		sum += Math.log(d + 6);
		sum += Math.log(d + 7);
		sum += Math.log(d + 8);
		sum += Math.log(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("log standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = 0.001;d < 10;d += 0.000001) {
		sum += log(d + 0);
		sum += log(d + 1);
		sum += log(d + 2);
		sum += log(d + 3);
		sum += log(d + 4);
		sum += log(d + 5);
		sum += log(d + 6);
		sum += log(d + 7);
		sum += log(d + 8);
		sum += log(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("log sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += Math.exp(d + 0);
		sum += Math.exp(d + 1);
		sum += Math.exp(d + 2);
		sum += Math.exp(d + 3);
		sum += Math.exp(d + 4);
		sum += Math.exp(d + 5);
		sum += Math.exp(d + 6);
		sum += Math.exp(d + 7);
		sum += Math.exp(d + 8);
		sum += Math.exp(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("exp standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += exp(d + 0);
		sum += exp(d + 1);
		sum += exp(d + 2);
		sum += exp(d + 3);
		sum += exp(d + 4);
		sum += exp(d + 5);
		sum += exp(d + 6);
		sum += exp(d + 7);
		sum += exp(d + 8);
		sum += exp(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("exp sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double y = 0;y < 100;y += 0.05) {
		for(double x = 0.001;x < 100;x += 0.05) {
		    sum += Math.pow(x + 0, y);
		    sum += Math.pow(x + 1, y);
		    sum += Math.pow(x + 2, y);
		    sum += Math.pow(x + 3, y);
		    sum += Math.pow(x + 4, y);
		    sum += Math.pow(x + 5, y);
		    sum += Math.pow(x + 6, y);
		    sum += Math.pow(x + 7, y);
		    sum += Math.pow(x + 8, y);
		    sum += Math.pow(x + 9, y);
		}
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("pow standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double y = 0;y < 100;y += 0.05) {
		for(double x = 0.001;x < 100;x += 0.05) {
		    sum += pow(x + 0, y);
		    sum += pow(x + 1, y);
		    sum += pow(x + 2, y);
		    sum += pow(x + 3, y);
		    sum += pow(x + 4, y);
		    sum += pow(x + 5, y);
		    sum += pow(x + 6, y);
		    sum += pow(x + 7, y);
		    sum += pow(x + 8, y);
		    sum += pow(x + 9, y);
		}
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("pow sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += Math.sinh(d + 0);
		sum += Math.sinh(d + 1);
		sum += Math.sinh(d + 2);
		sum += Math.sinh(d + 3);
		sum += Math.sinh(d + 4);
		sum += Math.sinh(d + 5);
		sum += Math.sinh(d + 6);
		sum += Math.sinh(d + 7);
		sum += Math.sinh(d + 8);
		sum += Math.sinh(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("sinh standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += sinh(d + 0);
		sum += sinh(d + 1);
		sum += sinh(d + 2);
		sum += sinh(d + 3);
		sum += sinh(d + 4);
		sum += sinh(d + 5);
		sum += sinh(d + 6);
		sum += sinh(d + 7);
		sum += sinh(d + 8);
		sum += sinh(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("sinh sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = 0;d < 10;d += 0.000005) {
		sum += Math.cosh(d + 0);
		sum += Math.cosh(d + 1);
		sum += Math.cosh(d + 2);
		sum += Math.cosh(d + 3);
		sum += Math.cosh(d + 4);
		sum += Math.cosh(d + 5);
		sum += Math.cosh(d + 6);
		sum += Math.cosh(d + 7);
		sum += Math.cosh(d + 8);
		sum += Math.cosh(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("cosh standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = 0;d < 10;d += 0.000005) {
		sum += cosh(d + 0);
		sum += cosh(d + 1);
		sum += cosh(d + 2);
		sum += cosh(d + 3);
		sum += cosh(d + 4);
		sum += cosh(d + 5);
		sum += cosh(d + 6);
		sum += cosh(d + 7);
		sum += cosh(d + 8);
		sum += cosh(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("cosh sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += Math.tanh(d + 0);
		sum += Math.tanh(d + 1);
		sum += Math.tanh(d + 2);
		sum += Math.tanh(d + 3);
		sum += Math.tanh(d + 4);
		sum += Math.tanh(d + 5);
		sum += Math.tanh(d + 6);
		sum += Math.tanh(d + 7);
		sum += Math.tanh(d + 8);
		sum += Math.tanh(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("tanh standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += tanh(d + 0);
		sum += tanh(d + 1);
		sum += tanh(d + 2);
		sum += tanh(d + 3);
		sum += tanh(d + 4);
		sum += tanh(d + 5);
		sum += tanh(d + 6);
		sum += tanh(d + 7);
		sum += tanh(d + 8);
		sum += tanh(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("tanh sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = 0;d < 10;d += 0.000005) {
		sum += Math.sqrt(d + 0);
		sum += Math.sqrt(d + 1);
		sum += Math.sqrt(d + 2);
		sum += Math.sqrt(d + 3);
		sum += Math.sqrt(d + 4);
		sum += Math.sqrt(d + 5);
		sum += Math.sqrt(d + 6);
		sum += Math.sqrt(d + 7);
		sum += Math.sqrt(d + 8);
		sum += Math.sqrt(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("sqrt standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = 0;d < 10;d += 0.000005) {
		sum += sqrt(d + 0);
		sum += sqrt(d + 1);
		sum += sqrt(d + 2);
		sum += sqrt(d + 3);
		sum += sqrt(d + 4);
		sum += sqrt(d + 5);
		sum += sqrt(d + 6);
		sum += sqrt(d + 7);
		sum += sqrt(d + 8);
		sum += sqrt(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("sqrt sleef ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += Math.cbrt(d + 0);
		sum += Math.cbrt(d + 1);
		sum += Math.cbrt(d + 2);
		sum += Math.cbrt(d + 3);
		sum += Math.cbrt(d + 4);
		sum += Math.cbrt(d + 5);
		sum += Math.cbrt(d + 6);
		sum += Math.cbrt(d + 7);
		sum += Math.cbrt(d + 8);
		sum += Math.cbrt(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("cbrt standard library ... " + (finish - start));

	    total += sum;
	}

	{
	    long start = System.currentTimeMillis();

	    double sum = 0;
	    for(double d = -10;d < 10;d += 0.00001) {
		sum += cbrt(d + 0);
		sum += cbrt(d + 1);
		sum += cbrt(d + 2);
		sum += cbrt(d + 3);
		sum += cbrt(d + 4);
		sum += cbrt(d + 5);
		sum += cbrt(d + 6);
		sum += cbrt(d + 7);
		sum += cbrt(d + 8);
		sum += cbrt(d + 9);
	    }

	    long finish = System.currentTimeMillis();

	    System.out.println("cbrt sleef ... " + (finish - start));

	    total += sum;
	}

	System.out.println();
	System.out.println("A meaningless value ... " + total);
    }
}
