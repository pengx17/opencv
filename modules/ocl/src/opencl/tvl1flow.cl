/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jin Ma jin@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

__kernel void centeredGradientKernel(__global const float* src, int src_col, int src_row, int src_step, 
                                     __global float* dx, __global float* dy, int dx_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if((x < src_col)&&(y < src_row))
    {
        int src_x1 = (x + 1) < (src_col -1)? (x + 1) : (src_col - 1);
        int src_x2 = (x - 1) > 0 ? (x -1) : 0;

        dx[y * dx_step+ x] = 0.5f * (src[y * src_step + src_x1] - src[y * src_step+ src_x2]);

        int src_y1 = (y+1) < (src_row - 1) ? (y + 1) : (src_row - 1);
        int src_y2 = (y - 1) > 0 ? (y - 1) : 0;
        dy[y * dx_step+ x] = 0.5f * (src[src_y1 * src_step + x] - src[src_y2 * src_step+ x]);
    }

}

float bicubicCoeff(float x_)
{

    float x = fabs(x_);
    if (x <= 1.0f)
    {
        return x * x * (1.5f * x - 2.5f) + 1.0f;
    }
    else if (x < 2.0f)
    {
        return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
    }
    else
    {
        return 0.0f;
    }

}

__kernel void warpBackwardKernel(__global const float* I0, int I0_step, int I0_col, int I0_row,
                                 __global const float* I1, __global const float* I1x, __global const float* I1y,  
                                 __global const float2* u1, int u1_step, 
                                 __global float* I1w,
                                 __global float* I1wx,
                                 __global float* I1wy,
                                 __global float* grad,
                                 __global float* rho,
                                 int I1w_step,
                                 int u1_offset_x,
                                 int u1_offset_y,
                                 int I1_step,
                                 int I1x_step)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if(x < I0_col &&y < I0_row)
    {
        const float2 uVal = u1[(y + u1_offset_y) * u1_step + x + u1_offset_x];

        const float2 wx = (float2)(x + uVal.x, y + uVal.y);

        const int xmin = ceil(wx.x - 2.0f);
        const int xmax = floor(wx.x + 2.0f);

        const int ymin = ceil(wx.y - 2.0f);
        const int ymax = floor(wx.y + 2.0f);

        float4 sum  = (float4)(0.0f);
        float4 sumx = (float4)(0.0f);
        float4 sumy = (float4)(0.0f);
        float4 wsum = (float4)(0.0f);

        for (int cy = ymin; cy <= ymax; ++cy)
        {
            const int cx = xmin;
            float4 w;
            const float wy = bicubicCoeff(wx.y - cy);
            w.x = bicubicCoeff(wx.x - cx)     * wy;
            w.y = bicubicCoeff(wx.x - cx - 1) * wy;
            w.z = bicubicCoeff(wx.x - cx - 2) * wy;
            w.w = bicubicCoeff(wx.x - cx - 3) * wy;

            const int clamped_cy = clamp(cy, 0, I0_row - 1);
            int4 offset_I = (int4)(clamped_cy * I1_step), offset_Ixy = (int4)(clamped_cy * I1x_step);
            int4 h_offset = (int4)(clamp(cx, 0, I0_col - 1), clamp(cx + 1, 0, I0_col - 1), clamp(cx + 2, 0, I0_col - 1), clamp(cx + 3, 0, I0_col - 1));

            offset_I += h_offset;
            offset_Ixy += h_offset;

            float4 I1_val, I1x_val, I1y_val;

            if(offset_I.w - offset_I.x == 4)
            {
                I1_val  = vload4(0, I1  + offset_I.x);
                I1x_val = vload4(0, I1x + offset_Ixy.x);
                I1y_val = vload4(0, I1y + offset_Ixy.x);
            }
            else
            {
                I1_val.x  = I1[offset_I.x];
                I1_val.y  = I1[offset_I.y];
                I1_val.z  = I1[offset_I.z];
                I1_val.w  = I1[offset_I.w];

                I1x_val.x = I1x[offset_Ixy.x];
                I1x_val.y = I1x[offset_Ixy.y];
                I1x_val.z = I1x[offset_Ixy.z];
                I1x_val.w = I1x[offset_Ixy.w];                
                
                I1y_val.x = I1y[offset_Ixy.x];
                I1y_val.y = I1y[offset_Ixy.y];
                I1y_val.z = I1y[offset_Ixy.z];
                I1y_val.w = I1y[offset_Ixy.w];
            }
            sum  += w * I1_val;
            sumx += w * I1x_val;
            sumy += w * I1y_val;
            wsum += w;
        }
        sum.w  += sum.x  + sum.y  + sum.z;
        sumx.w += sumx.x + sumx.y + sumx.z;
        sumy.w += sumy.x + sumy.y + sumy.z;
        wsum.w += wsum.x + wsum.y + wsum.z;

        const float coeff = 1.0f / wsum.w;

        const float I1wVal  = sum.w * coeff;
        const float I1wxVal = sumx.w * coeff;
        const float I1wyVal = sumy.w * coeff;

        I1w[y * I1w_step + x]  = I1wVal;
        I1wx[y * I1w_step + x] = I1wxVal;
        I1wy[y * I1w_step + x] = I1wyVal;

        const float Ix2 = I1wxVal * I1wxVal;
        const float Iy2 = I1wyVal * I1wyVal;

        grad[y * I1w_step + x] = Ix2 + Iy2;

        const float I0Val = I0[y * I0_step + x];
        rho[y * I1w_step + x] = I1wVal - I1wxVal * uVal.x - I1wyVal * uVal.y - I0Val;
    }
}
__kernel void estimateDualVariablesKernel(__global const float2* u, int u_col, int u_row, int u_step, 
                                          __global float4* p, int p_step, 
                                          const float taut,
                                          int u_offset_x,
                                          int u_offset_y)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if(x < u_col && y < u_row)
    {
        int src_x1 = (x + 1) < (u_col - 1) ? (x + 1) : (u_col - 1);
        const float2 u1x = u[(y + u_offset_y) * u_step + src_x1 + u_offset_x] - u[(y + u_offset_y) * u_step + x + u_offset_x];

        int src_y1 = (y + 1) < (u_row - 1) ? (y + 1) : (u_row - 1);
        const float2 u1y = u[(src_y1 + u_offset_y) * u_step + x + u_offset_x] - u[(y + u_offset_y) * u_step + x + u_offset_x];

        const float g1 = hypot(u1x.x, u1y.x);
        const float g2 = hypot(u1x.y, u1y.y);

        const float ng1 = 1.0f + taut * g1;
        const float ng2 = 1.0f + taut * g2;

        float4 p_temp = p[mad24(y, p_step, x)];

        p[mad24(y, p_step, x)] = (float4)((p_temp.x + taut * u1x.x) / ng1,
                                          (p_temp.y + taut * u1y.x) / ng1,
                                          (p_temp.z + taut * u1x.y) / ng2,
                                          (p_temp.w + taut * u1y.y) / ng2);
    }

}

float2 divergence(__global const float4* v, int y, int x, int v1_step)
{

    if (x > 0 && y > 0)
    {
        const float v1x = v[mad24(y, v1_step, x)].x - v[mad24(y, v1_step, x - 1)].x;
        const float v1y = v[mad24(y, v1_step, x)].y - v[mad24(y - 1, v1_step, x)].y;

        const float v2x = v[mad24(y, v1_step, x)].z - v[mad24(y, v1_step, x - 1)].z;
        const float v2y = v[mad24(y, v1_step, x)].w - v[mad24(y - 1, v1_step, x)].w;

        return (float2)(v1x + v1y, v2x + v2y);
    }
    else
    {
        if (y > 0)
        {
            const float v1x = v[y * v1_step].x + v[y * v1_step].y - v[(y - 1) * v1_step].y;
            const float v2x = v[y * v1_step].z + v[y * v1_step].w - v[(y - 1) * v1_step].w;
            return (float2)(v1x, v2x);
        }
        else
        {
            if (x > 0)
            {
                const float v1x = v[x].x - v[x - 1].x + v[x].y;
                const float v2x = v[x].z - v[x - 1].z + v[x].w;
                return (float2)(v1x, v2x);
            }
            else
            {
                const float v1x = v[0].x + v[0].y;
                const float v2x = v[0].z + v[0].w;
                return (float2)(v1x, v2x);
            }
        }
    }

}
__kernel void estimateUKernel(__global const float* I1wx, int I1wx_col, int I1wx_row, int I1wx_step,
                              __global const float* I1wy,
                              __global const float* grad, 
                              __global const float* rho_c,
                              __global const float4* p,
                              __global float2* u, int u1_step, 
                              __global float* error, const float l_t, const float theta,
                              int u_offset_x,
                              int u_offset_y,
                              char calc_error)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < I1wx_col && y < I1wx_row)
    {
        const float I1wxVal = I1wx[mad24(y, I1wx_step, x)];
        const float I1wyVal = I1wy[mad24(y, I1wx_step, x)];
        const float gradVal = grad[mad24(y, I1wx_step, x)];
        const float2 u1OldVal = u[mad24(y + u_offset_y, u1_step, x + u_offset_x)];

        const float rho = rho_c[mad24(y, I1wx_step, x)] + (I1wxVal * u1OldVal.x + I1wyVal * u1OldVal.y);

        float2 d1 = (float2)(l_t * I1wxVal, l_t * I1wyVal);

        if (rho < -l_t * gradVal)
        {
            d1 = d1;
        }
        else if (rho > l_t * gradVal)
        {
            d1 = -d1;
        }
        else if (gradVal > 1.192092896e-07f)
        {
            const float fi = -rho / gradVal;
            d1 = (float2)(fi * I1wxVal, fi * I1wyVal);
        }

        const float2 v = u1OldVal + d1;

        const float2 div_p = divergence(p, y, x, I1wx_step);

        const float2 u1NewVal = v + theta * div_p;

        u[(y + u_offset_y) * u1_step + x + u_offset_x] = u1NewVal;

        if(calc_error)
        {
            const float2 n = (u1OldVal - u1NewVal) * (u1OldVal - u1NewVal);
            error[y * I1wx_step + x] = n.x + n.y;
        }
    }

}
