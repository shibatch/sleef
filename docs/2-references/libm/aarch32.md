---
layout: default
title: AArch32
parent: Single & Double Precision
grand_parent: References
permalink: /2-references/libm/aarch32
---

<h1>Single & Double Precision Math library reference (AArch32)</h1>

<h2>Table of contents</h2>

<ul class="circle">
  <li><a href="#datatypes">Data types</a></li>
  <li><a href="#trig">Trigonometric functions</a></li>
  <li><a href="#pow">Power, exponential, and logarithmic functions</a></li>
  <li><a href="#invtrig">Inverse trigonometric functions</a></li>
  <li><a href="#hyp">Hyperbolic functions and inverse hyperbolic functions</a></li>
  <li><a href="#eg">Error and gamma functions</a></li>
  <li><a href="#nearint">Nearest integer functions</a></li>
  <li><a href="#other">Other functions</a></li>
</ul>

<h2 id="datatypes">Data types for AArch32 architecture</h2>

<p class="funcname"><b class="type">Sleef_float32x4_t_2</b></p>

<p class="header">Description</p>

<p class="noindent">
<b class="type">Sleef_float32x4_t_2</b> is a data type for storing two <b class="type">float32x4_t</b> values,
which is defined in sleef.h as follows:
</p>

<pre class="white">typedef struct {
  float32x4_t x, y;
} Sleef_float32x4_t_2;
</pre>


<h2 id="trig">Trigonometric Functions</h2>

<p class="funcname">Vectorized single precision sine function with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_sinf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_sinf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_sinf_u10"><b class="func">Sleef_sinf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision sine function with 3.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_sinf4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_sinf4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_sinf_u35"><b class="func">Sleef_sinf_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision cosine function with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_cosf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_cosf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_cosf_u10"><b class="func">Sleef_cosf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision cosine function with 3.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_cosf4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_cosf4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_cosf_u35"><b class="func">Sleef_cosf_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision combined sine and cosine function with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">Sleef_float32x4_t_2</b> <b class="func">Sleef_sincosf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">Sleef_float32x4_t_2</b> <b class="func">Sleef_sincosf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_sincosf_u10"><b class="func">Sleef_sincosf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision combined sine and cosine function with 3.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">Sleef_float32x4_t_2</b> <b class="func">Sleef_sincosf4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">Sleef_float32x4_t_2</b> <b class="func">Sleef_sincosf4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_sincosf_u35"><b class="func">Sleef_sincosf_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision sine function with 0.506 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_sinpif4_u05</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_sinpif4_u05neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_sinpif_u05"><b class="func">Sleef_sinpif_u05</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision cosine function with 0.506 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_cospif4_u05</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_cospif4_u05neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_cospif_u05"><b class="func">Sleef_cospif_u05</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision combined sine and cosine function with 0.506 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">Sleef_float32x4_t_2</b> <b class="func">Sleef_sincospif4_u05</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">Sleef_float32x4_t_2</b> <b class="func">Sleef_sincospif4_u05neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_sincospif_u05"><b class="func">Sleef_sincospif_u05</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision combined sine and cosine function with 3.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">Sleef_float32x4_t_2</b> <b class="func">Sleef_sincospif4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">Sleef_float32x4_t_2</b> <b class="func">Sleef_sincospif4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_sincospif_u35"><b class="func">Sleef_sincospif_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision tangent function with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_tanf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_tanf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_tanf_u10"><b class="func">Sleef_tanf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision tangent function with 3.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_tanf4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_tanf4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_tanf_u35"><b class="func">Sleef_tanf_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<h2 id="pow">Power, exponential, and logarithmic function</h2>

<p class="funcname">Vectorized single precision power function with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_powf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_powf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_powf_u10"><b class="func">Sleef_powf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision natural logarithmic function with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_logf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_logf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_logf_u10"><b class="func">Sleef_logf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision natural logarithmic function with 3.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_logf4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_logf4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_logf_u35"><b class="func">Sleef_logf_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision base-10 logarithmic function with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_log10f4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_log10f4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_log10f_u10"><b class="func">Sleef_log10f_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision base-2 logarithmic function with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_log2f4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_log2f4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_log2f_u10"><b class="func">Sleef_log2f_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision logarithm of one plus argument with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_log1pf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_log1pf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_log1pf_u10"><b class="func">Sleef_log1pf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision base-<i>e</i> exponential function function with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_expf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_expf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_expf_u10"><b class="func">Sleef_expf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision base-<i>2</i> exponential function function with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_exp2f4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_exp2f4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_exp2f_u10"><b class="func">Sleef_exp2f_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision base-10 exponential function function with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_exp10f4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_exp10f4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_exp10f_u10"><b class="func">Sleef_exp10f_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision base-<i>e</i> exponential function minus 1 with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_expm1f4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_expm1f4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_expm1f_u10"><b class="func">Sleef_expm1f_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision square root function with 0.5001 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_sqrtf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_sqrtf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_sqrtf_u05"><b class="func">Sleef_sqrtf_u05</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision square root function with 3.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_sqrtf4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_sqrtf4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_sqrtf_u35"><b class="func">Sleef_sqrtf_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision cubic root function with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_cbrtf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_cbrtf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_cbrtf_u10"><b class="func">Sleef_cbrtf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision cubic root function with 3.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_cbrtf4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_cbrtf4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_cbrtf_u35"><b class="func">Sleef_cbrtf_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision 2D Euclidian distance function with 0.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_hypotf4_u05</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_hypotf4_u05neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_hypotf_u05"><b class="func">Sleef_hypotf_u05</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision 2D Euclidian distance function with 3.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_hypotf4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_hypotf4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_hypotf_u35"><b class="func">Sleef_hypotf_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>


<h2 id="invtrig">Inverse Trigonometric Functions</h2>

<p class="funcname">Vectorized single precision arc sine function with 3.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_asinf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_asinf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_asinf_u10"><b class="func">Sleef_asinf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision arc sine function with 3.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_asinf4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_asinf4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_asinf_u35"><b class="func">Sleef_asinf_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision arc cosine function with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_acosf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_acosf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_acosf_u10"><b class="func">Sleef_acosf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision arc cosine function with 3.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_acosf4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_acosf4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_acosf_u35"><b class="func">Sleef_acosf_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision arc tangent function with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_atanf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_atanf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_atanf_u10"><b class="func">Sleef_atanf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision arc tangent function with 3.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_atanf4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_atanf4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_atanf_u35"><b class="func">Sleef_atanf_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision arc tangent function of two variables with 1.0 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_atan2f4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_atan2f4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_atan2f_u10"><b class="func">Sleef_atan2f_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision arc tangent function of two variables with 3.5 ULP error bound</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_atan2f4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_atan2f4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_atan2f_u35"><b class="func">Sleef_atan2f_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>



<h2 id="hyp">Hyperbolic function and inverse hyperbolic function</h2>

<p class="funcname">Vectorized single precision hyperbolic sine function</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_sinhf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_sinhf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_sinhf_u10"><b class="func">Sleef_sinhf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision hyperbolic sine function</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_sinhf4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_sinhf4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_sinhf_u35"><b class="func">Sleef_sinhf_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision hyperbolic cosine function</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_coshf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_coshf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_coshf_u10"><b class="func">Sleef_coshf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision hyperbolic cosine function</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_coshf4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_coshf4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_coshf_u35"><b class="func">Sleef_coshf_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision hyperbolic tangent function</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_tanhf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_tanhf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_tanhf_u10"><b class="func">Sleef_tanhf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision hyperbolic tangent function</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_tanhf4_u35</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_tanhf4_u35neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_tanhf_u35"><b class="func">Sleef_tanhf_u35</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision inverse hyperbolic sine function</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_asinhf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_asinhf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_asinhf_u10"><b class="func">Sleef_asinhf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision inverse hyperbolic cosine function</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_acoshf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_acoshf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_acoshf_u10"><b class="func">Sleef_acoshf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision inverse hyperbolic tangent function</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_atanhf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_atanhf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_atanhf_u10"><b class="func">Sleef_atanhf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>


<h2 id="eg">Error and gamma function</h2>

<p class="funcname">Vectorized single precision error function</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_erff4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_erff4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_erff_u10"><b class="func">Sleef_erff_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision complementary error function</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_erfcf4_u15</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_erfcf4_u15neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_erfcf_u15"><b class="func">Sleef_erfcf_u15</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision gamma function</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_tgammaf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_tgammaf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_tgammaf_u10"><b class="func">Sleef_tgammaf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision log gamma function</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_lgammaf4_u10</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_lgammaf4_u10neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_lgammaf_u10"><b class="func">Sleef_lgammaf_u10</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>


<h2 id="nearint">Nearest integer function</h2>

<p class="funcname">Vectorized single precision function for rounding to integer towards zero</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_truncf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_truncf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_truncf"><b class="func">Sleef_truncf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision function for rounding to integer towards negative infinity</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_floorf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_floorf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_floorf"><b class="func">Sleef_floorf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision function for rounding to integer towards positive infinity</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_ceilf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_ceilf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_ceilf"><b class="func">Sleef_ceilf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision function for rounding to nearest integer</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_roundf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_roundf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_roundf"><b class="func">Sleef_roundf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision function for rounding to nearest integer</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_rintf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_rintf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_rintf"><b class="func">Sleef_rintf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>


<h2 id="other">Other function</h2>

<p class="funcname">Vectorized single precision function for fused multiply-accumulation</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_fmaf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>, <b class="type">float32x4_t</b> <i class="var">c</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_fmaf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>, <b class="type">float32x4_t</b> <i class="var">c</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_fmaf"><b class="func">Sleef_fmaf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>

<p class="funcname">Vectorized single precision FP remainder</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_fmodf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_fmodf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_fmodf"><b class="func">Sleef_fmodf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>

<p class="funcname">Vectorized single precision FP remainder</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_remainderf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_remainderf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_remainderf"><b class="func">Sleef_remainderf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision function for obtaining fractional component of an FP number</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_frfrexpf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_frfrexpf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_frfrexpf"><b class="func">Sleef_frfrexpf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>

<p class="funcname">Vectorized single precision signed integral and fractional values</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">Sleef_float32x4_t_2</b> <b class="func">Sleef_modff4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">Sleef_float32x4_t_2</b> <b class="func">Sleef_modff4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_modff"><b class="func">Sleef_modff</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision function for calculating the absolute value</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_fabsf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_fabsf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_fabsf"><b class="func">Sleef_fabsf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision function for copying signs</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_copysignf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_copysignf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_copysignf"><b class="func">Sleef_copysignf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision function for determining maximum of two values</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_fmaxf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_fmaxf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_fmaxf"><b class="func">Sleef_fmaxf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision function for determining minimum of two values</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_fminf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_fminf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_fminf"><b class="func">Sleef_fminf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision function to calculate positive difference of two values</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_fdimf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_fdimf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_fdimf"><b class="func">Sleef_fdimf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>

<hr/>
<p class="funcname">Vectorized single precision function for obtaining the next representable FP value</p>

<p class="header">Synopsis</p>

<p class="synopsis">
#include &lt;sleef.h&gt;<br/>
<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_nextafterf4</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<b class="type">float32x4_t</b> <b class="func">Sleef_nextafterf4_neon</b>(<b class="type">float32x4_t</b> <i class="var">a</i>, <b class="type">float32x4_t</b> <i class="var">b</i>);<br/>
<br/>
<span class="normal">Link with</span> -lsleef.
</p>

<p class="header">Description</p>

<p class="noindent">
These are the vectorized functions of <a href="../libm#Sleef_nextafterf"><b class="func">Sleef_nextafterf</b></a>. This function may be less accurate than the scalar function since AArch32 NEON is not IEEE 754-compliant.
</p>
