package org.naokishibata.sleef;

/**
 * FastMath class is a Java implementation of the <a
 * href="http://freecode.com/projects/sleef">SLEEF</a>
 * library. Some of the methods can be used as substitutions of the
 * corresponding methods in Math class. They have slightly less
 * accuracy, and some methods are faster compared to those methods in
 * Math class. Please note that the methods in the standard Math class
 * are JNI methods, and the SLEEF library is specialized for SIMD
 * operations.
 */
public class FastMath {
    public static double E = Math.E;
    public static double PI = Math.PI;

    public static double abs(double a) { return Math.abs(a); }
    public static float abs(float a) { return Math.abs(a); }
    public static int abs(int a) { return Math.abs(a); }
    public static long abs(long a) { return Math.abs(a); }

    public static double ceil(double a) { return Math.ceil(a); }
    public static double floor(double a) { return Math.floor(a); }

    //

    static double upper(double d) {
	long l = Double.doubleToRawLongBits(d);
	return Double.longBitsToDouble(l & 0xfffffffff8000000L);
    }

    static double mla(double x, double y, double z) { return x * y + z; }

    static double mulsign(double x, double y) { return Math.copySign(1, y) * x; }

    //

    /**
       Returns the absolute value of the argument
    */
    public static double fabs(double d) { return Math.copySign(d, 1); }

    /**
       Returns the larger value of the two arguments. The result is
       undefined if denormal numbers are given.
    */
    public static double max(double x, double y) { return x > y ? x : y; }

    /**
       Checks if the argument is a NaN or not.
    */
    public static boolean isnan(double d) { return d != d; }

    /**
       Checks if the argument is either positive infinity or negative infinity.
    */
    public static boolean isinf(double d) { return fabs(d) == Double.POSITIVE_INFINITY; }

    static boolean ispinf(double d) { return d == Double.POSITIVE_INFINITY; }
    static boolean isminf(double d) { return d == Double.NEGATIVE_INFINITY; }

    /**
       Returns the integer value that is closest to the argument. The
       result is undefined if a denormal number is given.
    */
    public static double rint(double x) { return x < 0 ? (int)(x - 0.5) : (int)(x + 0.5); }

    /**
       Returns the result of multiplying the floating-point number x
       by 2 raised to the power q
    */
    public static double ldexp(double x, int q) {
	int m = q >> 31;
	m = (((m + q) >> 9) - m) << 7;
	q = q - (m << 2);
	m += 0x3ff;
	m = m < 0     ? 0     : m;
	m = m > 0x7ff ? 0x7ff : m;
	double u = Double.longBitsToDouble(((long)m) << 52);
	x = x * u * u * u * u;
	u = Double.longBitsToDouble(((long)(q + 0x3ff)) << 52);
	return x * u;
    }

    static double pow2i(int q) {
	return Double.longBitsToDouble(((long)(q + 0x3ff)) << 52);
    }

    static int ilogbp1(double d) {
	boolean m = d < 4.9090934652977266E-91;
	d = m ? 2.037035976334486E90 * d : d;
	int q = (int)(Double.doubleToRawLongBits(d) >> 52) & 0x7ff;
	q = m ? q - (300 + 0x03fe) : q - 0x03fe;
	return q;
    }

    /**
       Returns the exponent part of their argument as a signed integer
    */
    public static int ilogb(double d) {
	int e = ilogbp1(fabs(d)) - 1;
	e = d == 0 ? -2147483648 : e;
	e = d == Double.POSITIVE_INFINITY || d == Double.NEGATIVE_INFINITY ? 2147483647 : e;
	return e;
    }

    //

    static boolean cmpDenorm(double x, double y) {
	if (isnan(x) && isnan(y)) return true;
	if (x == Double.POSITIVE_INFINITY && y == Double.POSITIVE_INFINITY) return true;
	if (x == Double.NEGATIVE_INFINITY && y == Double.NEGATIVE_INFINITY) return true;
	if (!isnan(x) && !isnan(y) && !isinf(x) && !isinf(y)) return true;
	return false;
    }

    /**
       Checks if the argument is +0.
    */
    public static boolean isPlusZero(double x) { return x == 0 && Math.copySign(1, x) == 1; }

    /**
       Checks if the argument is -0.
    */
    public static boolean isMinusZero(double x) { return x == 0 && Math.copySign(1, x) == -1; }

    static double sign(double d) { return Math.copySign(1, d); }

    //

    /**
       This class represents a vector of two double values.
    */
    public static class double2 {
	public double x, y;
	public double2() {}
	public double2(double x, double y) { this.x = x; this.y = y; }

	public String toString() {
	    return "(double2:" + x + " + " + y + ")";
	}
    }

    static double2 ddnormalize_d2_d2(double2 t) {
	double2 s = new double2();

	s.x = t.x + t.y;
	s.y = t.x - s.x + t.y;

	return s;
    }

    static double2 ddscale_d2_d2_d(double2 d, double s) {
	double2 r = new double2();

	r.x = d.x * s;
	r.y = d.y * s;

	return r;
    }

    static double2 ddadd2_d2_d_d(double x, double y) {
	double2 r = new double2();

	r.x = x + y;
	double v = r.x - x;
	r.y = (x - (r.x - v)) + (y - v);

	return r;
    }

    static double2 ddadd_d2_d2_d(double2 x, double y) {
	// |x| >= |y|

	double2 r = new double2();

	//assert(isnan(x.x) || isnan(y) || fabs(x.x) >= fabs(y));

	r.x = x.x + y;
	r.y = x.x - r.x + y + x.y;

	return r;
    }

    static double2 ddadd2_d2_d2_d(double2 x, double y) {
	// |x| >= |y|

	double2 r = new double2();

	r.x  = x.x + y;
	double v = r.x - x.x;
	r.y = (x.x - (r.x - v)) + (y - v);
	r.y += x.y;

	return r;
    }

    static double2 ddadd_d2_d_d2(double x, double2 y) {
	// |x| >= |y|

	double2 r = new double2();

	//assert(isnan(x) || isnan(y.x) || fabs(x) >= fabs(y.x));

	r.x = x + y.x;
	r.y = x - r.x + y.x + y.y;

	return r;
    }

    static double2 ddadd_d2_d2_d2(double2 x, double2 y) {
	// |x| >= |y|

	double2 r = new double2();

	//assert(isnan(x.x) || isinf(x.x) || isnan(y.x) || isinf(y.x) || fabs(x.x) >= fabs(y.x)) : "x.x = " + x.x + ", y.x = " + y.x;

	r.x = x.x + y.x;
	r.y = x.x - r.x + y.x + x.y + y.y;

	return r;
    }

    static double2 ddadd2_d2_d2_d2(double2 x, double2 y) {
	double2 r = new double2();

	r.x  = x.x + y.x;
	double v = r.x - x.x;
	r.y = (x.x - (r.x - v)) + (y.x - v);
	r.y += x.y + y.y;

	return r;
    }

    static double2 ddsub_d2_d2_d2(double2 x, double2 y) {
	// |x| >= |y|

	double2 r = new double2();

	r.x = x.x - y.x;
	r.y = x.x - r.x - y.x + x.y - y.y;

	return r;
    }

    static double2 dddiv_d2_d2_d2(double2 n, double2 d) {
	double t = 1.0 / d.x;
	double dh  = upper(d.x), dl  = d.x - dh;
	double th  = upper(t  ), tl  = t   - th;
	double nhh = upper(n.x), nhl = n.x - nhh;

	double2 q = new double2();

	q.x = n.x * t;

	double u = -q.x + nhh * th + nhh * tl + nhl * th + nhl * tl +
	    q.x * (1 - dh * th - dh * tl - dl * th - dl * tl);

	q.y = t * (n.y - q.x * d.y) + u;

	return q;
    }

    static double2 ddmul_d2_d_d(double x, double y) {
	double xh = upper(x), xl = x - xh;
	double yh = upper(y), yl = y - yh;
	double2 r = new double2();

	r.x = x * y;
	r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl;

	return r;
    }

    static double2 ddmul_d2_d2_d(double2 x, double y) {
	double xh = upper(x.x), xl = x.x - xh;
	double yh = upper(y  ), yl = y   - yh;
	double2 r = new double2();

	r.x = x.x * y;
	r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl + x.y * y;

	return r;
    }

    static double2 ddmul_d2_d2_d2(double2 x, double2 y) {
	double xh = upper(x.x), xl = x.x - xh;
	double yh = upper(y.x), yl = y.x - yh;
	double2 r = new double2();

	r.x = x.x * y.x;
	r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl + x.x * y.y + x.y * y.x;

	return r;
    }

    static double2 ddsqu_d2_d2(double2 x) {
	double xh = upper(x.x), xl = x.x - xh;
	double2 r = new double2();

	r.x = x.x * x.x;
	r.y = xh * xh - r.x + (xh + xh) * xl + xl * xl + x.x * (x.y + x.y);

	return r;
    }

    static double2 ddrec_d2_d(double d) {
	double t = 1.0 / d;
	double dh = upper(d), dl = d - dh;
	double th = upper(t), tl = t - th;
	double2 q = new double2();

	q.x = t;
	q.y = t * (1 - dh * th - dh * tl - dl * th - dl * tl);

	return q;
    }

    static double2 ddrec_d2_d2(double2 d) {
	double t = 1.0 / d.x;
	double dh = upper(d.x), dl = d.x - dh;
	double th = upper(t  ), tl = t   - th;
	double2 q = new double2();
	
	q.x = t;
	q.y = t * (1 - dh * th - dh * tl - dl * th - dl * tl - d.y * t);

	return q;
    }

    static double2 ddsqrt_d2_d2(double2 d) {
	double t = Math.sqrt(d.x + d.y);
	return ddscale_d2_d2_d(ddmul_d2_d2_d2(ddadd2_d2_d2_d2(d, ddmul_d2_d_d(t, t)), ddrec_d2_d(t)), 0.5);
    }

    //

    static double atan2k(double y, double x) {
	double s, t, u;
	int q = 0;

	if (x < 0) { x = -x; q = -2; }
	if (y > x) { t = x; x = y; y = -t; q += 1; }

	s = y / x;
	t = s * s;

	u = -1.88796008463073496563746e-05;
	u = u * t + (0.000209850076645816976906797);
	u = u * t + (-0.00110611831486672482563471);
	u = u * t + (0.00370026744188713119232403);
	u = u * t + (-0.00889896195887655491740809);
	u = u * t + (0.016599329773529201970117);
	u = u * t + (-0.0254517624932312641616861);
	u = u * t + (0.0337852580001353069993897);
	u = u * t + (-0.0407629191276836500001934);
	u = u * t + (0.0466667150077840625632675);
	u = u * t + (-0.0523674852303482457616113);
	u = u * t + (0.0587666392926673580854313);
	u = u * t + (-0.0666573579361080525984562);
	u = u * t + (0.0769219538311769618355029);
	u = u * t + (-0.090908995008245008229153);
	u = u * t + (0.111111105648261418443745);
	u = u * t + (-0.14285714266771329383765);
	u = u * t + (0.199999999996591265594148);
	u = u * t + (-0.333333333333311110369124);

	t = u * t * s + s;
	t = q * (Math.PI/2) + t;

	return t;
    }

    /**
       This method calculates the arc tangent of y/x in radians, using
       the signs of the two arguments to determine the quadrant of the
       result. The results may have maximum error of 2 ulps.
    */
    public static double atan2(double y, double x) {
	double r = atan2k(fabs(y), x);

	r = mulsign(r, x);
	if (isinf(x) || x == 0) r = Math.PI/2 - (isinf(x) ? (sign(x) * (Math.PI  /2)) : 0);
	if (isinf(y)          ) r = Math.PI/2 - (isinf(x) ? (sign(x) * (Math.PI*1/4)) : 0);
	if (            y == 0) r = (sign(x) == -1 ? Math.PI : 0);

	return isnan(x) || isnan(y) ? Double.NaN : mulsign(r, y);
    }

    /**
       This method calculates the arc sine of x in radians. The return
       value is in the range [-pi/2, pi/2]. The results may have
       maximum error of 3 ulps.
    */
    public static double asin(double d) {
	return mulsign(atan2k(fabs(d), Math.sqrt((1+d)*(1-d))), d);
    }

    /**
       This method calculates the arc cosine of x in radians. The
       return value is in the range [0, pi]. The results may have
       maximum error of 3 ulps.
    */
    public static double acos(double d) {
	return mulsign(atan2k(Math.sqrt((1+d)*(1-d)), fabs(d)), d) + (d < 0 ? Math.PI : 0);
    }

    /**
       Returns the arc tangent of an angle. The results may have
       maximum error of 2 ulps.
    */
    public static double atan(double s) {
	double t, u;
	int q = 0;

	if (s < 0) { s = -s; q = 2; }
	if (s > 1) { s = 1.0 / s; q |= 1; }

	t = s * s;

	u = -1.88796008463073496563746e-05;
	u = u * t + (0.000209850076645816976906797);
	u = u * t + (-0.00110611831486672482563471);
	u = u * t + (0.00370026744188713119232403);
	u = u * t + (-0.00889896195887655491740809);
	u = u * t + (0.016599329773529201970117);
	u = u * t + (-0.0254517624932312641616861);
	u = u * t + (0.0337852580001353069993897);
	u = u * t + (-0.0407629191276836500001934);
	u = u * t + (0.0466667150077840625632675);
	u = u * t + (-0.0523674852303482457616113);
	u = u * t + (0.0587666392926673580854313);
	u = u * t + (-0.0666573579361080525984562);
	u = u * t + (0.0769219538311769618355029);
	u = u * t + (-0.090908995008245008229153);
	u = u * t + (0.111111105648261418443745);
	u = u * t + (-0.14285714266771329383765);
	u = u * t + (0.199999999996591265594148);
	u = u * t + (-0.333333333333311110369124);

	t = s + s * (t * u);

	if ((q & 1) != 0) t = 1.570796326794896557998982 - t;
	if ((q & 2) != 0) t = -t;

	return t;
    }

    private static final double PI4_A = 0.78539816290140151978;
    private static final double PI4_B = 4.9604678871439933374e-10;
    private static final double PI4_C = 1.1258708853173288931e-18;
    private static final double PI4_D = 1.7607799325916000908e-27;

    private static final double M_1_PI = 0.3183098861837906715377675267450287;

    /**
       Returns the trigonometric sine of an angle. The results may
       have maximum error of 2 ulps.
    */
    public static double sin(double d) {
	int q;
	double u, s;

	u = d * M_1_PI;
	q = (int)(u < 0 ? u - 0.5 : u + 0.5);

	d = mla(q, -PI4_A*4, d);
	d = mla(q, -PI4_B*4, d);
	d = mla(q, -PI4_C*4, d);
	d = mla(q, -PI4_D*4, d);

	if ((q & 1) != 0) d = -d;

	s = d * d;

	u = -7.97255955009037868891952e-18;
	u = mla(u, s, 2.81009972710863200091251e-15);
	u = mla(u, s, -7.64712219118158833288484e-13);
	u = mla(u, s, 1.60590430605664501629054e-10);
	u = mla(u, s, -2.50521083763502045810755e-08);
	u = mla(u, s, 2.75573192239198747630416e-06);
	u = mla(u, s, -0.000198412698412696162806809);
	u = mla(u, s, 0.00833333333333332974823815);
	u = mla(u, s, -0.166666666666666657414808);

	u = mla(s, u * d, d);

	return u;
    }

    /**
       Returns the trigonometric cosine of an angle. The results may
       have maximum error of 2 ulps.
    */
    public static double cos(double d) {
	int q;
	double u, s;

	q = 1 + 2*(int)rint(d * M_1_PI - 0.5);

	d = mla(q, -PI4_A*2, d);
	d = mla(q, -PI4_B*2, d);
	d = mla(q, -PI4_C*2, d);
	d = mla(q, -PI4_D*2, d);

	if ((q & 2) == 0) d = -d;

	s = d * d;

	u = -7.97255955009037868891952e-18;
	u = mla(u, s, 2.81009972710863200091251e-15);
	u = mla(u, s, -7.64712219118158833288484e-13);
	u = mla(u, s, 1.60590430605664501629054e-10);
	u = mla(u, s, -2.50521083763502045810755e-08);
	u = mla(u, s, 2.75573192239198747630416e-06);
	u = mla(u, s, -0.000198412698412696162806809);
	u = mla(u, s, 0.00833333333333332974823815);
	u = mla(u, s, -0.166666666666666657414808);

	u = mla(s, u * d, d);

	return u;
    }

    /**
       Returns the trigonometric sine and cosine of an angle at a
       time. The sine and cosine of an argument is returned by the x
       and y field of the return value, respectively. The results may
       have maximum error of 2 ulps.
    */
    public static double2 sincos(double d) {
	int q;
	double u, s, t;
        double2 r = new double2();

	q = (int)rint(d * (2 * M_1_PI));

	s = d;

	s = mla(-q, PI4_A*2, s);
	s = mla(-q, PI4_B*2, s);
	s = mla(-q, PI4_C*2, s);
	s = mla(-q, PI4_D*2, s);

	t = s;

	s = s * s;

	u = 1.58938307283228937328511e-10;
	u = mla(u, s, -2.50506943502539773349318e-08);
	u = mla(u, s, 2.75573131776846360512547e-06);
	u = mla(u, s, -0.000198412698278911770864914);
	u = mla(u, s, 0.0083333333333191845961746);
	u = mla(u, s, -0.166666666666666130709393);
	u = u * s * t;

	r.x = t + u;

	u = -1.13615350239097429531523e-11;
	u = mla(u, s, 2.08757471207040055479366e-09);
	u = mla(u, s, -2.75573144028847567498567e-07);
	u = mla(u, s, 2.48015872890001867311915e-05);
	u = mla(u, s, -0.00138888888888714019282329);
	u = mla(u, s, 0.0416666666666665519592062);
	u = mla(u, s, -0.5);

	r.y = u * s + 1;

	if ((q & 1) != 0) { s = r.y; r.y = r.x; r.x = s; }
	if ((q & 2) != 0) { r.x = -r.x; }
	if (((q+1) & 2) != 0) { r.y = -r.y; }

	if (isinf(d)) { r.x = r.y = Double.NaN; }

	return r;
    }

    /**
       Returns the trigonometric tangent of an angle. The results may
       have maximum error of 3 ulps.
    */
    public static double tan(double d) {
	int q;
	double u, s, x;

	q = (int)rint(d * (2 * M_1_PI));

	x = mla(q, -PI4_A*2, d);
	x = mla(q, -PI4_B*2, x);
	x = mla(q, -PI4_C*2, x);
	x = mla(q, -PI4_D*2, x);

	s = x * x;

	if ((q & 1) != 0) x = -x;

	u = 1.01419718511083373224408e-05;
	u = mla(u, s, -2.59519791585924697698614e-05);
	u = mla(u, s, 5.23388081915899855325186e-05);
	u = mla(u, s, -3.05033014433946488225616e-05);
	u = mla(u, s, 7.14707504084242744267497e-05);
	u = mla(u, s, 8.09674518280159187045078e-05);
	u = mla(u, s, 0.000244884931879331847054404);
	u = mla(u, s, 0.000588505168743587154904506);
	u = mla(u, s, 0.00145612788922812427978848);
	u = mla(u, s, 0.00359208743836906619142924);
	u = mla(u, s, 0.00886323944362401618113356);
	u = mla(u, s, 0.0218694882853846389592078);
	u = mla(u, s, 0.0539682539781298417636002);
	u = mla(u, s, 0.133333333333125941821962);
	u = mla(u, s, 0.333333333333334980164153);

	u = mla(s, u * x, x);

	if ((q & 1) != 0) u = 1.0 / u;

	if (isinf(d)) u = Double.NaN;

	return u;
    }

    //

    private static final double L2U = .69314718055966295651160180568695068359375;
    private static final double L2L = .28235290563031577122588448175013436025525412068e-12;
    private static final double R_LN2 = 1.442695040888963407359924681001892137426645954152985934135449406931;

    /**
       Returns the natural logarithm of the argument. The results may
       have maximum error of 3 ulps.
    */
    public static double log(double d) {
	double x, x2, t, m;
	int e, i;

	e = ilogbp1(d * 0.7071);
	m = ldexp(d, -e);

	x = (m-1) / (m+1);
	x2 = x * x;

	t = 0.148197055177935105296783;
	t = mla(t, x2, 0.153108178020442575739679);
	t = mla(t, x2, 0.181837339521549679055568);
	t = mla(t, x2, 0.22222194152736701733275);
	t = mla(t, x2, 0.285714288030134544449368);
	t = mla(t, x2, 0.399999999989941956712869);
	t = mla(t, x2, 0.666666666666685503450651);
	t = mla(t, x2, 2);

	x = x * t + 0.693147180559945286226764 * e;

	if (ispinf(d)) x = Double.POSITIVE_INFINITY;
	if (d < 0) x = Double.NaN;
	if (d == 0) x = Double.NEGATIVE_INFINITY;

	return x;
    }

    /**
       Returns the value of e raised to the power of the argument. The
       results may have maximum error of 1 ulps.
    */
    public static double exp(double d) {
	int q = (int)rint(d * R_LN2);
	double s, u;

	s = mla(q, -L2U, d);
	s = mla(q, -L2L, s);

	u = 2.08860621107283687536341e-09;
	u = mla(u, s, 2.51112930892876518610661e-08);
	u = mla(u, s, 2.75573911234900471893338e-07);
	u = mla(u, s, 2.75572362911928827629423e-06);
	u = mla(u, s, 2.4801587159235472998791e-05);
	u = mla(u, s, 0.000198412698960509205564975);
	u = mla(u, s, 0.00138888888889774492207962);
	u = mla(u, s, 0.00833333333331652721664984);
	u = mla(u, s, 0.0416666666666665047591422);
	u = mla(u, s, 0.166666666666666851703837);
	u = mla(u, s, 0.5);

	u = s * s * u + s + 1;
	u = ldexp(u, q);

	if (isminf(d)) u = 0;

	return u;
    }

    static double2 logk(double d) {
	double2 x, x2;
	double m, t;
	int e;

	e = ilogbp1(d * 0.7071);
	m = ldexp(d, -e);

	x = dddiv_d2_d2_d2(ddadd2_d2_d_d(-1, m), ddadd2_d2_d_d(1, m));
	x2 = ddsqu_d2_d2(x);

	t = 0.134601987501262130076155;
	t = mla(t, x2.x, 0.132248509032032670243288);
	t = mla(t, x2.x, 0.153883458318096079652524);
	t = mla(t, x2.x, 0.181817427573705403298686);
	t = mla(t, x2.x, 0.222222231326187414840781);
	t = mla(t, x2.x, 0.285714285651261412873718);
	t = mla(t, x2.x, 0.400000000000222439910458);
	t = mla(t, x2.x, 0.666666666666666371239645);

	return ddadd2_d2_d2_d2(ddmul_d2_d2_d(new double2(0.693147180559945286226764, 2.319046813846299558417771e-17), e),
			       ddadd2_d2_d2_d2(ddscale_d2_d2_d(x, 2), ddmul_d2_d2_d(ddmul_d2_d2_d2(x2, x), t)));
    }

    static double expk(double2 d) {
	int q = (int)rint((d.x + d.y) * R_LN2);
	double2 s, t;
	double u;

	s = ddadd2_d2_d2_d(d, -q * L2U);
	s = ddadd2_d2_d2_d(s, -q * L2L);

	s = ddnormalize_d2_d2(s);

	u = 2.51069683420950419527139e-08;
	u = mla(u, s.x, 2.76286166770270649116855e-07);
	u = mla(u, s.x, 2.75572496725023574143864e-06);
	u = mla(u, s.x, 2.48014973989819794114153e-05);
	u = mla(u, s.x, 0.000198412698809069797676111);
	u = mla(u, s.x, 0.0013888888939977128960529);
	u = mla(u, s.x, 0.00833333333332371417601081);
	u = mla(u, s.x, 0.0416666666665409524128449);
	u = mla(u, s.x, 0.166666666666666740681535);
	u = mla(u, s.x, 0.500000000000000999200722);

	t = ddadd_d2_d2_d2(s, ddmul_d2_d2_d(ddsqu_d2_d2(s), u));

	t = ddadd_d2_d_d2(1, t);

	return ldexp(t.x + t.y, q);
    }

    /**
       Returns the value of the first argument raised to the power of
       the second argument. The results may have maximum error of 1
       ulps.
    */
    public static double pow(double x, double y) {
	boolean yisint = (int)y == y;
	boolean yisodd = (1 & (int)y) != 0 && yisint;

	double result = expk(ddmul_d2_d2_d(logk(fabs(x)), y));

	result = isnan(result) ? Double.POSITIVE_INFINITY : result;
	result *=  (x >= 0 ? 1 : (!yisint ? Double.NaN : (yisodd ? -1 : 1)));

	double efx = mulsign(fabs(x) - 1, y);
	if (isinf(y)) result = efx < 0 ? 0.0 : (efx == 0 ? 1.0 : Double.POSITIVE_INFINITY);
	if (isinf(x) || x == 0) result = (yisodd ? sign(x) : 1) * ((x == 0 ? -y : y) < 0 ? 0 : Double.POSITIVE_INFINITY);
	if (isnan(x) || isnan(y)) result = Double.NaN;
	if (y == 0 || x == 1) result = 1;

	return result;
    }

    static double2 expk2(double2 d) {
	int q = (int)rint((d.x + d.y) * R_LN2);
	double2 s, t;
	double u;

	s = ddadd2_d2_d2_d(d, q * -L2U);
	s = ddadd2_d2_d2_d(s, q * -L2L);

	s = ddnormalize_d2_d2(s);

	u = 2.51069683420950419527139e-08;
	u = mla(u, s.x, 2.76286166770270649116855e-07);
	u = mla(u, s.x, 2.75572496725023574143864e-06);
	u = mla(u, s.x, 2.48014973989819794114153e-05);
	u = mla(u, s.x, 0.000198412698809069797676111);
	u = mla(u, s.x, 0.0013888888939977128960529);
	u = mla(u, s.x, 0.00833333333332371417601081);
	u = mla(u, s.x, 0.0416666666665409524128449);
	u = mla(u, s.x, 0.166666666666666740681535);
	u = mla(u, s.x, 0.500000000000000999200722);

	t = ddadd_d2_d2_d2(s, ddmul_d2_d2_d(ddsqu_d2_d2(s), u));

	t = ddadd_d2_d_d2(1, t);
	return ddscale_d2_d2_d(t, pow2i(q));
    }

    /**
       Returns the hyperbolic sine of x. The results may have maximum
       error of 2 ulps.
    */
    public static double sinh(double x) {
	double y = fabs(x);
	double2 d = expk2(new double2(y, 0));
	d = ddsub_d2_d2_d2(d, ddrec_d2_d2(d));
	y = (d.x + d.y) * 0.5;

	y = abs(x) > 710 ? Double.POSITIVE_INFINITY : y;
	y = isnan(y) ? Double.POSITIVE_INFINITY : y;
	y = mulsign(y, x);
	y = isnan(x) ? Double.NaN : y;

	return y;
    }

    /**
       Returns the hyperbolic cosine of x. The results may have
       maximum error of 2 ulps.
    */
    public static double cosh(double x) {
	double y = fabs(x);
	double2 d = expk2(new double2(y, 0));
	d = ddadd_d2_d2_d2(d, ddrec_d2_d2(d));
	y = (d.x + d.y) * 0.5;

	y = abs(x) > 710 ? Double.POSITIVE_INFINITY : y;
	y = isnan(y) ? Double.POSITIVE_INFINITY : y;
	y = isnan(x) ? Double.NaN : y;

	return y;
    }

    /**
       Returns the hyperbolic tangent of x. The results may have
       maximum error of 2 ulps.
    */
    public static double tanh(double x) {
	double y = fabs(x);
	double2 d = expk2(new double2(y, 0));
	double2 e = dddiv_d2_d2_d2(new double2(1, 0), d);
	d = dddiv_d2_d2_d2(ddadd2_d2_d2_d2(d, ddscale_d2_d2_d(e, -1)), ddadd2_d2_d2_d2(d, e));
	y = d.x + d.y;

	y = abs(x) > 18.714973875 ? 1.0 : y;
	y = isnan(y) ? 1.0 : y;
	y = mulsign(y, x);
	y = isnan(x) ? Double.NaN : y;

	return y;
    }

    static double2 logk2(double2 d) {
	double2 x, x2, m;
	double t;
	int e;

	e = ilogbp1(d.x * 0.7071);
	m = ddscale_d2_d2_d(d, pow2i(-e));

	x = dddiv_d2_d2_d2(ddadd2_d2_d2_d(m, -1), ddadd2_d2_d2_d(m, 1));
	x2 = ddsqu_d2_d2(x);

	t = 0.134601987501262130076155;
	t = mla(t, x2.x, 0.132248509032032670243288);
	t = mla(t, x2.x, 0.153883458318096079652524);
	t = mla(t, x2.x, 0.181817427573705403298686);
	t = mla(t, x2.x, 0.222222231326187414840781);
	t = mla(t, x2.x, 0.285714285651261412873718);
	t = mla(t, x2.x, 0.400000000000222439910458);
	t = mla(t, x2.x, 0.666666666666666371239645);

	return ddadd2_d2_d2_d2(ddmul_d2_d2_d(new double2(0.693147180559945286226764, 2.319046813846299558417771e-17), e),
			       ddadd2_d2_d2_d2(ddscale_d2_d2_d(x, 2), ddmul_d2_d2_d(ddmul_d2_d2_d2(x2, x), t)));
    }

    /**
       Returns the inverse hyperbolic sine of x. The results may have
       maximum error of 2 ulps.
    */
    public static double asinh(double x) {
	double y = fabs(x);
	double2 d = logk2(ddadd2_d2_d2_d(ddsqrt_d2_d2(ddadd2_d2_d2_d(ddmul_d2_d_d(y, y),  1)), y));
	y = d.x + d.y;

	y = isinf(x) || isnan(y) ? Double.POSITIVE_INFINITY : y;
	y = mulsign(y, x);
	y = isnan(x) ? Double.NaN : y;

	return y;
    }

    /**
       Returns the inverse hyperbolic cosine of x. The results may
       have maximum error of 2 ulps.
    */
    public static double acosh(double x) {
	double2 d = logk2(ddadd2_d2_d2_d(ddsqrt_d2_d2(ddadd2_d2_d2_d(ddmul_d2_d_d(x, x), -1)), x));
	double y = d.x + d.y;

	y = isinf(x) || isnan(y) ? Double.POSITIVE_INFINITY : y;
	y = x == 1.0 ? 0.0 : y;
	y = x < 1.0 ? Double.NaN : y;
	y = isnan(x) ? Double.NaN : y;

	return y;
    }

    /**
       Returns the inverse hyperbolic tangent of x. The results may
       have maximum error of 2 ulps.
    */
    public static double atanh(double x) {
	double y = fabs(x);
	double2 d = logk2(dddiv_d2_d2_d2(ddadd2_d2_d_d(1, y), ddadd2_d2_d_d(1, -y)));
	y = y > 1.0 ? Double.NaN : (y == 1.0 ? Double.POSITIVE_INFINITY : (d.x + d.y) * 0.5);

	y = isinf(x) || isnan(y) ? Double.NaN : y;
	y = mulsign(y, x);
	y = isnan(x) ? Double.NaN : y;

	return y;
    }

    /**
       This function performs a fused multiply-accumulate
       operation. This function computes x*y+z, with a single
       rounding. This implementation gives the exact result unless an
       overflow occurs.
    */
    public static double fma(double x, double y, double z) {
	double xh = Double.longBitsToDouble((Double.doubleToRawLongBits(x) + 0x4000000) & 0xfffffffff8000000L), xl = x - xh;
	double yh = Double.longBitsToDouble((Double.doubleToRawLongBits(y) + 0x4000000) & 0xfffffffff8000000L), yl = y - yh;

	double h = x * y;
	double l = xh * yh - h + xl * yh + xh * yl + xl * yl;

	double h2, l2, v;

	h2 = h + z;
	v = h2 - h;
	l2 = (h - (h2 - v)) + (z - v) + l;

	return h2 + l2;
    }

    /**
       This function returns the square root of the argument.  This
       implementation gives the exact result(less than or equal to 0.5
       ulp of error).
    */
    public static double sqrt(double d) {
	double q = 1;

	if (d < 8.636168555094445E-78) {
	    d *= 1.157920892373162E77;
	    q = 2.9387358770557188E-39;
	}

	// http://en.wikipedia.org/wiki/Fast_inverse_square_root
	double x = Double.longBitsToDouble(0x5fe6ec85e7de30daL - (Double.doubleToRawLongBits(d + 1e-320) >> 1));

	x = x * (1.5 - 0.5 * d * x * x);
	x = x * (1.5 - 0.5 * d * x * x);
	x = x * (1.5 - 0.5 * d * x * x);

	x = fma(d * x, d * x, -d) * (x * -0.5) + d * x;

	return d == Double.POSITIVE_INFINITY ? Double.POSITIVE_INFINITY : x * q;
    }

    /**
       This function returns the cube root of the argument. The
       results may have maximum error of 2 ulps.
    */
    public static double cbrt(double d) {
	double x, y, q = 1.0;
	int e, r;

	e = ilogbp1(d);
	d = ldexp(d, -e);
	r = (e + 6144) % 3;
	q = (r == 1) ? 1.2599210498948731647672106 : q;
	q = (r == 2) ? 1.5874010519681994747517056 : q;
	q = ldexp(q, (e + 6144) / 3 - 2048);

	q = mulsign(q, d);
	d = fabs(d);

	x = -0.640245898480692909870982;
	x = x * d + 2.96155103020039511818595;
	x = x * d + -5.73353060922947843636166;
	x = x * d + 6.03990368989458747961407;
	x = x * d + -3.85841935510444988821632;
	x = x * d + 2.2307275302496609725722;

	y = x * x; y = y * y; x -= (d * y - x) * (1.0 / 3.0);
	y = d * x * x;
	y = (y - (2.0 / 3.0) * y * (y * x - 1)) * q;

	return y;
    }

    /**
       Returns the value of 2 raised to the power of the argument. The
       results may have maximum error of 1 ulp.
    */
    public static double exp2(double a) {
	double u = expk(ddmul_d2_d2_d(new double2(0.69314718055994528623, 2.3190468138462995584e-17), a));
	if (a > 1023) u = Double.POSITIVE_INFINITY;
	if (isminf(a)) u = 0;
	return u;
    }

    /**
       Returns the value of 10 raised to the power of the
       argument. The results may have maximum error of 1 ulp.
    */
    public static double exp10(double a) {
	double u = expk(ddmul_d2_d2_d(new double2(2.3025850929940459011, -2.1707562233822493508e-16), a));
	if (a > 308) u = Double.POSITIVE_INFINITY;
	if (isminf(a)) u = 0;
	return u;
    }

    /**
       Returns a value equivalent to exp(a)-1. The result is accurate
       even when the value of a is close to zero. The results may have
       maximum error of 1 ulp.
    */
    public static double expm1(double a) {
	double2 d = ddadd2_d2_d2_d(expk2(new double2(a, 0)), -1.0);
	double x = d.x + d.y;
	if (a > 700) x = Double.POSITIVE_INFINITY;
	if (a < -0.36043653389117156089696070315825181539851971360337e+2) x = -1;
	return x;
    }

    /**
       Returns the base 10 logarithm of the argument. The results may
       have maximum error of 1 ulp.
    */
    public static double log10(double a) {
	double2 d = ddmul_d2_d2_d2(logk(a), new double2(0.43429448190325176116, 6.6494347733425473126e-17));
	double x = d.x + d.y;

	if (ispinf(a)) x = Double.POSITIVE_INFINITY;
	if (a < 0) x = Double.NaN;
	if (a == 0) x = -Double.POSITIVE_INFINITY;

	return x;
    }

    /**
       Returns a value equivalent to log(1+a). The result is accurate
       even when the value of a is close to zero. The results may have
       maximum error of 1 ulp.
    */
    public static double log1p(double a) {
	double2 d = logk2(ddadd2_d2_d_d(a, 1));
	double x = d.x + d.y;

	if (ispinf(a)) x = Double.POSITIVE_INFINITY;
	if (a < -1) x = Double.NaN;
	if (a == -1) x = -Double.POSITIVE_INFINITY;

	return x;
    }

    //

    /**
       This class represents a vector of two float values.
    */
    public static class float2 {
	public float x, y;
	public float2() {}
	public float2(float x, float y) { this.x = x; this.y = y; }

	public String toString() {
	    return "(float2:" + x + " + " + y + ")";
	}
    }

    private static final float PI4_Af = 0.78515625f;
    private static final float PI4_Bf = 0.00024187564849853515625f;
    private static final float PI4_Cf = 3.7747668102383613586e-08f;
    private static final float PI4_Df = 1.2816720341285448015e-12f;

    private static final float L2Uf = 0.693145751953125f;
    private static final float L2Lf = 1.428606765330187045e-06f;

    private static final float R_LN2f = 1.442695040888963407359924681001892137426645954152985934135449406931f;
    private static final float M_PIf = ((float)Math.PI);

    private static final float INFINITYf = Float.POSITIVE_INFINITY;
    private static final float NANf = Float.NaN;

    private static float mlaf(float x, float y, float z) { return x * y + z; }
    private static float mulsignf(float x, float y) { return (float)(Math.copySign(1, y) * x); }
    private static float signf(float d) { return (float)Math.copySign(1, d); }

    private static float sqrtf(float f) { return (float)Math.sqrt(f); }

    private static float fabsf(float d) { return (float)Math.copySign(d, 1); }
    private static float maxf(float x, float y) { return x > y ? x : y; }
    private static boolean isnanf(float d) { return d != d; }
    private static boolean isinff(float d) { return fabs(d) == Float.POSITIVE_INFINITY; }

    private static boolean ispinff(float d) { return d == Float.POSITIVE_INFINITY; }
    private static boolean isminff(float d) { return d == Float.NEGATIVE_INFINITY; }
    private static float rintf(float x) { return x < 0 ? (int)(x - 0.5f) : (int)(x + 0.5f); }

    static int floatToRawIntBits(float d) { return Float.floatToRawIntBits(d); }
    static float intBitsToFloat(int i) { return Float.intBitsToFloat(i); }

    static int ilogbp1f(float d) {
	boolean m = d < 5.421010862427522E-20f;
	d = m ? 1.8446744073709552E19f * d : d;
	int q = (floatToRawIntBits(d) >> 23) & 0xff;
	q = m ? q - (64 + 0x7e) : q - 0x7e;
	return q;
    }

    static float pow2if(int q) {
	return intBitsToFloat(((int)(q + 0x7f)) << 23);
    }

    public static float ldexpf(float x, int q) {
	float u;
	int m;
	m = q >> 31;
	m = (((m + q) >> 6) - m) << 4;
	q = q - (m << 2);
	m += 127;
	m = m <   0 ?   0 : m;
	m = m > 255 ? 255 : m;
	u = intBitsToFloat(((int)m) << 23);
	x = x * u * u * u * u;
	u = intBitsToFloat(((int)(q + 0x7f)) << 23);
	return x * u;
    }

    static float upperf(float d) {
	return intBitsToFloat(floatToRawIntBits(d) & 0xfffff000);
    }

    static float2 df(float h, float l) {
	float2 ret = new float2();
	ret.x = h; ret.y = l;
	return ret;
    }

    static float2 dfnormalize_f2_f2(float2 t) {
	float2 s = new float2();

	s.x = t.x + t.y;
	s.y = t.x - s.x + t.y;

	return s;
    }

    static float2 dfscale_f2_f2_f(float2 d, float s) {
	float2 r = new float2();

	r.x = d.x * s;
	r.y = d.y * s;

	return r;
    }

    static float2 dfadd2_f2_f_f(float x, float y) {
	float2 r = new float2();

	r.x = x + y;
	float v = r.x - x;
	r.y = (x - (r.x - v)) + (y - v);

	return r;
    }

    static float2 dfadd_f2_f2_f(float2 x, float y) {
	// |x| >= |y|

	float2 r = new float2();

	r.x = x.x + y;
	r.y = x.x - r.x + y + x.y;

	return r;
    }

    static float2 dfadd2_f2_f2_f(float2 x, float y) {
	// |x| >= |y|

	float2 r = new float2();

	r.x  = x.x + y;
	float v = r.x - x.x;
	r.y = (x.x - (r.x - v)) + (y - v);
	r.y += x.y;

	return r;
    }

    static float2 dfadd_f2_f_f2(float x, float2 y) {
	// |x| >= |y|

	float2 r = new float2();

	r.x = x + y.x;
	r.y = x - r.x + y.x + y.y;

	return r;
    }

    static float2 dfadd_f2_f2_f2(float2 x, float2 y) {
	// |x| >= |y|

	float2 r = new float2();

	r.x = x.x + y.x;
	r.y = x.x - r.x + y.x + x.y + y.y;

	return r;
    }

    static float2 dfadd2_f2_f2_f2(float2 x, float2 y) {
	float2 r = new float2();

	r.x  = x.x + y.x;
	float v = r.x - x.x;
	r.y = (x.x - (r.x - v)) + (y.x - v);
	r.y += x.y + y.y;

	return r;
    }

    static float2 dfsub_f2_f2_f2(float2 x, float2 y) {
	// |x| >= |y|

	float2 r = new float2();

	r.x = x.x - y.x;
	r.y = x.x - r.x - y.x + x.y - y.y;

	return r;
    }

    static float2 dfdiv_f2_f2_f2(float2 n, float2 d) {
	float t = 1.0f / d.x;
	float dh  = upperf(d.x), dl  = d.x - dh;
	float th  = upperf(t  ), tl  = t   - th;
	float nhh = upperf(n.x), nhl = n.x - nhh;

	float2 q = new float2();

	q.x = n.x * t;

	float u = -q.x + nhh * th + nhh * tl + nhl * th + nhl * tl +
	    q.x * (1 - dh * th - dh * tl - dl * th - dl * tl);

	q.y = t * (n.y - q.x * d.y) + u;

	return q;
    }

    static float2 dfmul_f2_f_f(float x, float y) {
	float xh = upperf(x), xl = x - xh;
	float yh = upperf(y), yl = y - yh;
	float2 r = new float2();

	r.x = x * y;
	r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl;

	return r;
    }

    static float2 dfmul_f2_f2_f(float2 x, float y) {
	float xh = upperf(x.x), xl = x.x - xh;
	float yh = upperf(y  ), yl = y   - yh;
	float2 r = new float2();

	r.x = x.x * y;
	r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl + x.y * y;

	return r;
    }

    static float2 dfmul_f2_f2_f2(float2 x, float2 y) {
	float xh = upperf(x.x), xl = x.x - xh;
	float yh = upperf(y.x), yl = y.x - yh;
	float2 r = new float2();

	r.x = x.x * y.x;
	r.y = xh * yh - r.x + xl * yh + xh * yl + xl * yl + x.x * y.y + x.y * y.x;

	return r;
    }

    static float2 dfsqu_f2_f2(float2 x) {
	float xh = upperf(x.x), xl = x.x - xh;
	float2 r = new float2();

	r.x = x.x * x.x;
	r.y = xh * xh - r.x + (xh + xh) * xl + xl * xl + x.x * (x.y + x.y);

	return r;
    }

    static float2 dfrec_f2_f(float d) {
	float t = 1.0f / d;
	float dh = upperf(d), dl = d - dh;
	float th = upperf(t), tl = t - th;
	float2 q = new float2();

	q.x = t;
	q.y = t * (1 - dh * th - dh * tl - dl * th - dl * tl);

	return q;
    }

    static float2 dfrec_f2_f2(float2 d) {
	float t = 1.0f / d.x;
	float dh = upperf(d.x), dl = d.x - dh;
	float th = upperf(t  ), tl = t   - th;
	float2 q = new float2();

	q.x = t;
	q.y = t * (1 - dh * th - dh * tl - dl * th - dl * tl - d.y * t);

	return q;
    }

    static float2 dfsqrt_f2_f2(float2 d) {
	float t = sqrtf(d.x + d.y);
	return dfscale_f2_f2_f(dfmul_f2_f2_f2(dfadd2_f2_f2_f2(d, dfmul_f2_f_f(t, t)), dfrec_f2_f(t)), 0.5f);
    }

    /**
       This function returns the cube root of the argument in single
       precision. The results may have maximum error of 2 ulps.
    */
    public static float cbrtf(float d) {
	float x, y, q = 1.0f;
	int e, r;

	e = ilogbp1f(d);
	d = ldexpf(d, -e);
	r = (e + 6144) % 3;
	q = (r == 1) ? 1.2599210498948731647672106f : q;
	q = (r == 2) ? 1.5874010519681994747517056f : q;
	q = ldexpf(q, (e + 6144) / 3 - 2048);

	q = mulsignf(q, d);
	d = fabsf(d);

	x = -0.601564466953277587890625f;
	x = mlaf(x, d, 2.8208892345428466796875f);
	x = mlaf(x, d, -5.532182216644287109375f);
	x = mlaf(x, d, 5.898262500762939453125f);
	x = mlaf(x, d, -3.8095417022705078125f);
	x = mlaf(x, d, 2.2241256237030029296875f);

	y = d * x * x;
	y = (y - (2.0f / 3.0f) * y * (y * x - 1.0f)) * q;

	return y;
    }

    /**
       Returns the trigonometric sine of an angle in single
       precision. The results may have maximum error of 3 ulps.
    */
    public static float sinf(float d) {
	int q;
	float u, s;

	q = (int)rintf(d * (float)M_1_PI);

	d = mlaf(q, -PI4_Af*4, d);
	d = mlaf(q, -PI4_Bf*4, d);
	d = mlaf(q, -PI4_Cf*4, d);
	d = mlaf(q, -PI4_Df*4, d);

	s = d * d;

	if ((q & 1) != 0) d = -d;

	u = 2.6083159809786593541503e-06f;
	u = mlaf(u, s, -0.0001981069071916863322258f);
	u = mlaf(u, s, 0.00833307858556509017944336f);
	u = mlaf(u, s, -0.166666597127914428710938f);

	u = mlaf(s, u * d, d);

	if (isinff(d)) { u = NANf; }

	return u;
    }

    /**
       Returns the trigonometric cosine of an angle in single
       precision. The results may have maximum error of 3 ulps.
    */
    public static float cosf(float d) {
	int q;
	float u, s;

	q = 1 + 2*(int)rintf(d * (float)M_1_PI - 0.5f);

	d = mlaf(q, -PI4_Af*2, d);
	d = mlaf(q, -PI4_Bf*2, d);
	d = mlaf(q, -PI4_Cf*2, d);
	d = mlaf(q, -PI4_Df*2, d);

	s = d * d;

	if ((q & 2) == 0) d = -d;

	u = 2.6083159809786593541503e-06f;
	u = mlaf(u, s, -0.0001981069071916863322258f);
	u = mlaf(u, s, 0.00833307858556509017944336f);
	u = mlaf(u, s, -0.166666597127914428710938f);

	u = mlaf(s, u * d, d);

	if (isinff(d)) { u = NANf; }

	return u;
    }

    /**
       Returns the trigonometric sine and cosine of an angle in single
       precision at a time. The sine and cosine of an argument is
       returned by the x and y field of the return value,
       respectively. The results may have maximum error of 3 ulps.
    */
    public static float2 sincosf(float d) {
	int q;
	float u, s, t;
	float2 r = new float2();

	q = (int)rintf(d * ((float)(2 * M_1_PI)));

	s = d;

	s = mlaf(q, -PI4_Af*2, s);
	s = mlaf(q, -PI4_Bf*2, s);
	s = mlaf(q, -PI4_Cf*2, s);
	s = mlaf(q, -PI4_Df*2, s);

	t = s;

	s = s * s;

	u = -0.000195169282960705459117889f;
	u = mlaf(u, s, 0.00833215750753879547119141f);
	u = mlaf(u, s, -0.166666537523269653320312f);
	u = u * s * t;

	r.x = t + u;

	u = -2.71811842367242206819355e-07f;
	u = mlaf(u, s, 2.47990446951007470488548e-05f);
	u = mlaf(u, s, -0.00138888787478208541870117f);
	u = mlaf(u, s, 0.0416666641831398010253906f);
	u = mlaf(u, s, -0.5f);

	r.y = u * s + 1;

	if ((q & 1) != 0) { s = r.y; r.y = r.x; r.x = s; }
	if ((q & 2) != 0) { r.x = -r.x; }
	if (((q+1) & 2) != 0) { r.y = -r.y; }

	if (isinff(d)) { r.x = r.y = NANf; }

	return r;
    }

    /**
       Returns the trigonometric tangent of an angle in single
       precision. The results may have maximum error of 4 ulps.
    */
    public static float tanf(float d) {
	int q;
	float u, s, x;

	q = (int)rintf(d * (float)(2 * M_1_PI));

	x = d;

	x = mlaf(q, -PI4_Af*2, x);
	x = mlaf(q, -PI4_Bf*2, x);
	x = mlaf(q, -PI4_Cf*2, x);
	x = mlaf(q, -PI4_Df*2, x);

	s = x * x;

	if ((q & 1) != 0) x = -x;

	u = 0.00927245803177356719970703f;
	u = mlaf(u, s, 0.00331984995864331722259521f);
	u = mlaf(u, s, 0.0242998078465461730957031f);
	u = mlaf(u, s, 0.0534495301544666290283203f);
	u = mlaf(u, s, 0.133383005857467651367188f);
	u = mlaf(u, s, 0.333331853151321411132812f);

	u = mlaf(s, u * x, x);

	if ((q & 1) != 0) u = 1.0f / u;

	if (isinff(d)) u = NANf;

	return u;
    }

    /**
       Returns the arc tangent of an angle in single precision. The
       results may have maximum error of 3 ulps.
    */
    public static float atanf(float s) {
	float t, u;
	int q = 0;

	if (s < 0) { s = -s; q = 2; }
	if (s > 1) { s = 1.0f / s; q |= 1; }

	t = s * s;

	u = 0.00282363896258175373077393f;
	u = mlaf(u, t, -0.0159569028764963150024414f);
	u = mlaf(u, t, 0.0425049886107444763183594f);
	u = mlaf(u, t, -0.0748900920152664184570312f);
	u = mlaf(u, t, 0.106347933411598205566406f);
	u = mlaf(u, t, -0.142027363181114196777344f);
	u = mlaf(u, t, 0.199926957488059997558594f);
	u = mlaf(u, t, -0.333331018686294555664062f);

	t = s + s * (t * u);

	if ((q & 1) != 0) t = 1.570796326794896557998982f - t;
	if ((q & 2) != 0) t = -t;

	return t;
    }

    private static float atan2kf(float y, float x) {
	float s, t, u;
	int q = 0;

	if (x < 0) { x = -x; q = -2; }
	if (y > x) { t = x; x = y; y = -t; q += 1; }

	s = y / x;
	t = s * s;

	u = 0.00282363896258175373077393f;
	u = mlaf(u, t, -0.0159569028764963150024414f);
	u = mlaf(u, t, 0.0425049886107444763183594f);
	u = mlaf(u, t, -0.0748900920152664184570312f);
	u = mlaf(u, t, 0.106347933411598205566406f);
	u = mlaf(u, t, -0.142027363181114196777344f);
	u = mlaf(u, t, 0.199926957488059997558594f);
	u = mlaf(u, t, -0.333331018686294555664062f);

	t = u * t * s + s;
	t = q * (float)(M_PIf/2) + t;

	return t;
    }

    /**
       This method calculates the arc tangent of y/x in single
       precision. It uses the signs of the two arguments to determine
       the quadrant of the result. The results may have maximum error
       of 3 ulps.
    */
    public static float atan2f(float y, float x) {
	float r = atan2kf(fabsf(y), x);

	r = mulsignf(r, x);
	if (isinff(x) || x == 0) r = M_PIf/2 - (isinff(x) ? (signf(x) * (float)(M_PIf  /2)) : 0);
	if (isinff(y)          ) r = M_PIf/2 - (isinff(x) ? (signf(x) * (float)(M_PIf*1/4)) : 0);
	if (              y == 0) r = (signf(x) == -1 ? M_PIf : 0);

	return isnanf(x) || isnanf(y) ? NANf : mulsignf(r, y);
    }

    /**
       This method calculates the arc sine of x in single
       precision. The results may have maximum error of 3 ulps.
    */
    public static float asinf(float d) {
	return mulsignf(atan2kf(fabsf(d), sqrtf((1.0f+d)*(1.0f-d))), d);
    }

    /**
       This method calculates the arc cosine of x in single
       precision. The results may have maximum error of 3 ulps.
    */
    public static float acosf(float d) {
	return mulsignf(atan2kf(sqrtf((1.0f+d)*(1.0f-d)), fabsf(d)), d) + (d < 0 ? (float)M_PIf : 0.0f);
    }

    /**
       Returns the natural logarithm of the argument in single
       precision. The results may have maximum error of 3 ulps.
    */
    public static float logf(float d) {
	float x, x2, t, m;
	int e;

	e = ilogbp1f(d * 0.7071f);
	m = ldexpf(d, -e);

	x = (m-1.0f) / (m+1.0f);
	x2 = x * x;

	t = 0.2371599674224853515625f;
	t = mlaf(t, x2, 0.285279005765914916992188f);
	t = mlaf(t, x2, 0.400005519390106201171875f);
	t = mlaf(t, x2, 0.666666567325592041015625f);
	t = mlaf(t, x2, 2.0f);

	x = x * t + 0.693147180559945286226764f * e;

	if (isinff(d)) x = INFINITYf;
	if (d < 0) x = NANf;
	if (d == 0) x = -INFINITYf;

	return x;
    }

    /**
       Returns the value of e raised to the power of the argument in
       single precision. The results may have maximum error of 1 ulps.
    */
    public static float expf(float d) {
	int q = (int)rintf(d * R_LN2f);
	float s, u;

	s = mlaf(q, -L2Uf, d);
	s = mlaf(q, -L2Lf, s);

	u = 0.00136324646882712841033936f;
	u = mlaf(u, s, 0.00836596917361021041870117f);
	u = mlaf(u, s, 0.0416710823774337768554688f);
	u = mlaf(u, s, 0.166665524244308471679688f);
	u = mlaf(u, s, 0.499999850988388061523438f);

	u = s * s * u + s + 1.0f;
	u = ldexpf(u, q);

	if (isminff(d)) u = 0;

	return u;
    }

    static float expkf(float2 d) {
	int q = (int)rintf((d.x + d.y) * R_LN2f);
	float2 s, t;
	float u;

	s = dfadd2_f2_f2_f(d, q * -L2Uf);
	s = dfadd2_f2_f2_f(s, q * -L2Lf);

	s = dfnormalize_f2_f2(s);

	u = 0.00136324646882712841033936f;
	u = mlaf(u, s.x, 0.00836596917361021041870117f);
	u = mlaf(u, s.x, 0.0416710823774337768554688f);
	u = mlaf(u, s.x, 0.166665524244308471679688f);
	u = mlaf(u, s.x, 0.499999850988388061523438f);

	t = dfadd_f2_f2_f2(s, dfmul_f2_f2_f(dfsqu_f2_f2(s), u));

	t = dfadd_f2_f_f2(1, t);
	return ldexpf(t.x + t.y, q);
    }

    static float2 logkf(float d) {
	float2 x, x2;
	float m, t;
	int e;

	e = ilogbp1f(d * 0.7071f);
	m = ldexpf(d, -e);

	x = dfdiv_f2_f2_f2(dfadd2_f2_f_f(-1, m), dfadd2_f2_f_f(1, m));
	x2 = dfsqu_f2_f2(x);

	t = 0.2371599674224853515625f;
	t = mlaf(t, x2.x, 0.285279005765914916992188f);
	t = mlaf(t, x2.x, 0.400005519390106201171875f);
	t = mlaf(t, x2.x, 0.666666567325592041015625f);

	return dfadd2_f2_f2_f2(dfmul_f2_f2_f(df(0.69314718246459960938f, -1.904654323148236017e-09f), e),
			       dfadd2_f2_f2_f2(dfscale_f2_f2_f(x, 2), dfmul_f2_f2_f(dfmul_f2_f2_f2(x2, x), t)));
    }

    public static float powf(float x, float y) {
	boolean yisint = (int)y == y;
	boolean yisodd = (1 & (int)y) != 0 && yisint;

	float result = expkf(dfmul_f2_f2_f(logkf(fabsf(x)), y));

	result = isnanf(result) ? INFINITYf : result;
	result *=  (x >= 0 ? 1 : (!yisint ? NANf : (yisodd ? -1 : 1)));

	float efx = mulsignf(fabsf(x) - 1, y);
	if (isinff(y)) result = efx < 0 ? 0.0f : (efx == 0 ? 1.0f : INFINITYf);
	if (isinff(x) || x == 0) result = (yisodd ? signf(x) : 1) * ((x == 0 ? -y : y) < 0 ? 0 : INFINITYf);
	if (isnanf(x) || isnanf(y)) result = NANf;
	if (y == 0 || x == 1) result = 1;

	return result;
    }

    static float2 expk2f(float2 d) {
	int q = (int)rintf((d.x + d.y) * R_LN2f);
	float2 s, t;
	float u;

	s = dfadd2_f2_f2_f(d, q * -L2Uf);
	s = dfadd2_f2_f2_f(s, q * -L2Lf);

	s = dfnormalize_f2_f2(s);

	u = 0.00136324646882712841033936f;
	u = mlaf(u, s.x, 0.00836596917361021041870117f);
	u = mlaf(u, s.x, 0.0416710823774337768554688f);
	u = mlaf(u, s.x, 0.166665524244308471679688f);
	u = mlaf(u, s.x, 0.499999850988388061523438f);

	t = dfadd_f2_f2_f2(s, dfmul_f2_f2_f(dfsqu_f2_f2(s), u));

	t = dfadd_f2_f_f2(1, t);
	return dfscale_f2_f2_f(t, pow2if(q));
    }

    public static float sinhf(float x) {
	float y = fabsf(x);
	float2 d = expk2f(df(y, 0));
	d = dfsub_f2_f2_f2(d, dfrec_f2_f2(d));
	y = (d.x + d.y) * 0.5f;

	y = fabsf(x) > 89 ? INFINITYf : y;
	y = isnanf(y) ? INFINITYf : y;
	y = mulsignf(y, x);
	y = isnanf(x) ? NANf : y;

	return y;
    }

    public static float coshf(float x) {
	float y = fabsf(x);
	float2 d = expk2f(df(y, 0));
	d = dfadd_f2_f2_f2(d, dfrec_f2_f2(d));
	y = (d.x + d.y) * 0.5f;

	y = fabsf(x) > 89 ? INFINITYf : y;
	y = isnanf(y) ? INFINITYf : y;
	y = isnanf(x) ? NANf : y;

	return y;
    }

    public static float tanhf(float x) {
	float y = fabsf(x);
	float2 d = expk2f(df(y, 0));
	float2 e = dfdiv_f2_f2_f2(df(1, 0), d);
	d = dfdiv_f2_f2_f2(dfadd2_f2_f2_f2(d, dfscale_f2_f2_f(e, -1)), dfadd2_f2_f2_f2(d, e));
	y = d.x + d.y;

	y = fabsf(x) > 8.664339742f ? 1.0f : y;
	y = isnanf(y) ? 1.0f : y;
	y = mulsignf(y, x);
	y = isnanf(x) ? NANf : y;

	return y;
    }

    static float2 logk2f(float2 d) {
	float2 x, x2, m;
	float t;
	int e;

	e = ilogbp1f(d.x * 0.7071f);
	m = dfscale_f2_f2_f(d, pow2if(-e));

	x = dfdiv_f2_f2_f2(dfadd2_f2_f2_f(m, -1), dfadd2_f2_f2_f(m, 1));
	x2 = dfsqu_f2_f2(x);

	t = 0.2371599674224853515625f;
	t = mlaf(t, x2.x, 0.285279005765914916992188f);
	t = mlaf(t, x2.x, 0.400005519390106201171875f);
	t = mlaf(t, x2.x, 0.666666567325592041015625f);

	return dfadd2_f2_f2_f2(dfmul_f2_f2_f(df(0.69314718246459960938f, -1.904654323148236017e-09f), e),
			       dfadd2_f2_f2_f2(dfscale_f2_f2_f(x, 2), dfmul_f2_f2_f(dfmul_f2_f2_f2(x2, x), t)));
    }

    public static float asinhf(float x) {
	float y = fabsf(x);
	float2 d = logk2f(dfadd2_f2_f2_f(dfsqrt_f2_f2(dfadd2_f2_f2_f(dfmul_f2_f_f(y, y),  1)), y));
	y = d.x + d.y;

	y = isinff(x) || isnanf(y) ? INFINITYf : y;
	y = mulsignf(y, x);
	y = isnanf(x) ? NANf : y;

	return y;
    }

    public static float acoshf(float x) {
	float2 d = logk2f(dfadd2_f2_f2_f(dfsqrt_f2_f2(dfadd2_f2_f2_f(dfmul_f2_f_f(x, x), -1)), x));
	float y = d.x + d.y;

	y = isinff(x) || isnanf(y) ? INFINITYf : y;
	y = x == 1.0f ? 0.0f : y;
	y = x < 1.0f ? NANf : y;
	y = isnanf(x) ? NANf : y;

	return y;
    }

    public static float atanhf(float x) {
	float y = fabsf(x);
	float2 d = logk2f(dfdiv_f2_f2_f2(dfadd2_f2_f_f(1, y), dfadd2_f2_f_f(1, -y)));
	y = y > 1.0 ? NANf : (y == 1.0 ? INFINITYf : (d.x + d.y) * 0.5f);

	y = isinff(x) || isnanf(y) ? NANf : y;
	y = mulsignf(y, x);
	y = isnanf(x) ? NANf : y;

	return y;
    }

    public static float exp2f(float a) {
	float u = expkf(dfmul_f2_f2_f(df(0.69314718246459960938f, -1.904654323148236017e-09f), a));
	if (ispinff(a)) u = INFINITYf;
	if (isminff(a)) u = 0;
	return u;
    }

    public static float exp10f(float a) {
	float u = expkf(dfmul_f2_f2_f(df(2.3025851249694824219f, -3.1975436520781386207e-08f), a));
	if (ispinff(a)) u = INFINITYf;
	if (isminff(a)) u = 0;
	return u;
    }

    public static float expm1f(float a) {
	float2 d = dfadd2_f2_f2_f(expk2f(df(a, 0)), -1.0f);
	float x = d.x + d.y;
	if (a > 88.0f) x = INFINITYf;
	if (a < -0.15942385152878742116596338793538061065739925620174e+2f) x = -1;
	return x;
    }

    public static float log10f(float a) {
	float2 d = dfmul_f2_f2_f2(logkf(a), df(0.43429449200630187988f, -1.0103050118726031315e-08f));
	float x = d.x + d.y;

	if (isinff(a)) x = INFINITYf;
	if (a < 0) x = NANf;
	if (a == 0) x = -INFINITYf;

	return x;
    }

    public static float log1pf(float a) {
	float2 d = logk2f(dfadd2_f2_f_f(a, 1));
	float x = d.x + d.y;

	if (isinff(a)) x = INFINITYf;
	if (a < -1) x = NANf;
	if (a == -1) x = -INFINITYf;

	return x;
    }
}
