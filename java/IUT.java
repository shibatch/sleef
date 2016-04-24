import java.io.*;

import org.naokishibata.sleef.*;

public class IUT {
    static long hexToLong(String s) {
	long ret = 0;
	for(int i=0;i<s.length();i++) {
	    char c = s.charAt(i);
	    ret <<= 4;
	    if ('0' <= c && c <= '9') ret += c - '0'; else ret += c - 'a' + 10;
	}
	return ret;
    }

    static String longToHex(long l) {
	if (l == 0) return "0";
	String str = "";
	while(l != 0) {
	    int d = (int)l & 0xf;
	    l = (l >>> 4) & 0x7fffffffffffffffL;
	    str = Character.forDigit(d, 16) + str;
	}
	return str;
    }

    public static void main(String[] args) throws Exception {
	LineNumberReader lnr = new LineNumberReader(new InputStreamReader(System.in));

	for(;;) {
	    String s = lnr.readLine();
	    if (s == null) break;

	    if (s.startsWith("atan2 ")) {
		String[] a = s.split(" ");
		long y = hexToLong(a[1]);
		long x = hexToLong(a[2]);
		double d = FastMath.atan2(Double.longBitsToDouble(y), Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("pow ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		long y = hexToLong(a[2]);
		double d = FastMath.pow(Double.longBitsToDouble(x), Double.longBitsToDouble(y));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("sincos ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		FastMath.double2 d2 = FastMath.sincos(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d2.x)) + " " + longToHex(Double.doubleToRawLongBits(d2.y)));
	    } else if (s.startsWith("sin ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.sin(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("cos ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.cos(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("tan ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.tan(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("asin ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.asin(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("acos ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.acos(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("atan ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.atan(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("log ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.log(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("exp ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.exp(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("sinh ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.sinh(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("cosh ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.cosh(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("tanh ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.tanh(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("asinh ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.asinh(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("acosh ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.acosh(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("atanh ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.atanh(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("sqrt ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.sqrt(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("cbrt ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.cbrt(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("exp2 ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.exp2(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("exp10 ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.exp10(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("expm1 ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.expm1(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("log10 ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.log10(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("log1p ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		double d = FastMath.log1p(Double.longBitsToDouble(x));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("ldexp ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]), y = hexToLong(a[2]);
		double d = FastMath.ldexp(Double.longBitsToDouble(x), (int)Double.longBitsToDouble(y));
		System.out.println(longToHex(Double.doubleToRawLongBits(d)));
	    } else if (s.startsWith("sinf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.sinf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("cosf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.cosf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("sincosf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		FastMath.float2 d2 = FastMath.sincosf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d2.x)) + " " + longToHex(Float.floatToRawIntBits(d2.y)));
	    } else if (s.startsWith("tanf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.tanf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("asinf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.asinf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("acosf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.acosf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("atanf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.atanf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("logf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.logf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("expf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.expf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("cbrtf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.cbrtf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("atan2f ")) {
		String[] a = s.split(" ");
		long y = hexToLong(a[1]);
		long x = hexToLong(a[2]);
		float d = FastMath.atan2f(Float.intBitsToFloat((int)y), Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("ldexpf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		long y = hexToLong(a[2]);
		float d = FastMath.ldexpf(Float.intBitsToFloat((int)x), (int)Float.intBitsToFloat((int)y));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("powf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		long y = hexToLong(a[2]);
		float d = FastMath.powf(Float.intBitsToFloat((int)x), Float.intBitsToFloat((int)y));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("sinhf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.sinhf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("coshf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.coshf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("tanhf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.tanhf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("asinhf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.asinhf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("acoshf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.acoshf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("atanhf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.atanhf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("exp2f ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.exp2f(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("exp10f ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.exp10f(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("expm1f ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.expm1f(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("log10f ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.log10f(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("log1pf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = FastMath.log1pf(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else if (s.startsWith("sqrtf ")) {
		String[] a = s.split(" ");
		long x = hexToLong(a[1]);
		float d = (float)Math.sqrt(Float.intBitsToFloat((int)x));
		System.out.println(longToHex(Float.floatToRawIntBits(d)));
	    } else {
		break;
	    }

	    System.out.flush();
	}
    }
}
