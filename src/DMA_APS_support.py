#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
from scipy.signal import savgol_filter
import os
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks
import matplotlib.ticker as ticker

from numpy import linspace, meshgrid
from scipy.interpolate import griddata

import matplotlib.font_manager
from matplotlib.ticker import AutoMinorLocator

import DMA_APS_inversion_tools
import Aerosol_tools

##---------------------------------------------------##
##                Contour plots                      ##
##---------------------------------------------------##
def Get_data_contour(data,z,column,reshape_DZ):
    D = data['Dp2'].values.copy()
    Z = data[z].values.copy()
    Img = data[column].values.copy()
    return Z,D,Img
def Show_countour(data1,
                     title,
                     espV1,
                     z = "z1",
                     column = "N_deconv",
                     cmap = 'viridis',
                     size_x = 14,
                     size_y = 7,
                    normalized=False,
                    reshape_DZ=False,
                    logscale=True,
                    logxscale=False,
                    logyscale=False,
                    zmax=400,
                    dmax=8000,
                    export=False):
    fig, ax1 = plt.subplots(num=None, figsize=(size_x, size_y), dpi=80, facecolor='w', edgecolor='k')
    
    Z1, D1, Img1 = Get_data_contour(data1,z,column,reshape_DZ)

    if(column == "N_deconv"):
        if(normalized):
            Img01 /= 1100
            Img02 /= 1100
            levels = np.array([1, 20, 50, 70, 100, 300, 500, 800, 1100]) /1100
        else:
            levels = np.array([1, 20, 50, 70, 100, 300, 500, 800, 1100])
    elif(column == "p"):
        levels = np.array([75, 100, 200, 300, 500, 600, 800, 900, 1100]) /60000
        #levels = np.array([0.0010, 0.0020, 0.0030, 0.0050, 0.0060, 0.0070, 0.0080, 0.01, 0.02])
    elif(column == "f1"):
        levels = np.linspace(0.005,0.3,100)

    Img1[Img1< np.min(levels)] = np.min(levels)

    cs1 = ax1.tricontourf(D1, Z1, Img1,
                          levels=levels,
                          cmap=cmap,
                          norm=LogNorm())
    
    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18)
    ax1.tick_params(direction='in', length=6, width=1, colors='k',
               grid_color='k', grid_alpha=0.5)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')

    ax1.set_xlabel('Particle diameter, $D_p$ (nm)', fontsize=18)
    ax1.set_ylabel('Elementary charges, z (-)', fontsize=18)
    
    ax1.set_ylim([np.min(Z1),zmax])
    ax1.set_xlim([np.min(D1),dmax])
    
    if(logxscale):
        ax1.set_xscale("log")
    if(logyscale):
        ax1.set_yscale("log")

    clb = plt.colorbar(cs1, ax=ax1, format=ticker.FuncFormatter(fmt))
    if(column == "N_deconv"):
        if(normalized):
            clb.ax.set_title('$f_1$ (-)',fontsize=18)
        else:
            clb.ax.set_title('N (#/cm$^3$)',fontsize=18)
    elif(column == "f1"):
        clb.ax.set_title('$f_1$ (-)',fontsize=18)
    elif(column == "p"):
        clb.ax.set_title('$f_1$ (-)',fontsize=18)
    plt.show()
    if(export):
        plt.savefig('Figures/'+title+'.png')
    return

def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi)
    X, Y = meshgrid(xi, yi)
    return X, Y, Z
def indiv_contourP(data,z,column,reshape_DZ):
    D = np.unique(data['Dp2'].values.copy())
    Z = np.unique(data[z].values.copy())
    Img0 = data[column].values.copy()

    D = list(dict.fromkeys(D))
    Z = list(dict.fromkeys(Z))
    ratio_ZD = len(D)/len(Z)

    if(reshape_DZ):
        Img = Img0.reshape(len(D),len(Z))
    else:
        Img = Img0.reshape(len(Z),len(D))
    return Z, D, Img,Img0
def fmt(x, pos):
    #format_float = "{:.2f}".format(x)
    a, b = '{:.1e}'.format(x).split('e')
    format_float = "{:.2f}".format(x)
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

##---------------------------------------------------##
## Fit peaks & analyze peaks
##---------------------------------------------------##



##---------------------------------------------------##
## Deconvolution
##---------------------------------------------------##
def Deconvolution(R, A, alpha):
    H = np.dot( A.transpose(), A) + alpha*np.identity(A.shape[1])
    rhs = np.dot( A.transpose(), R)
    return np.linalg.solve(H, rhs)
def Convolution(A, N):
    R_conv = np.zeros_like(N)
    for i in range(len(N)):
        R_conv[i] = np.sum(A[i,:] * N)
    return R_conv
def tDMA_convolution_matrix(V,Dpz,dma1,dma2,zn_max=6):
    z_n_vec = np.arange(1,zn_max)
    theta = len(V[:,0])
    #m = len(zDp[:,0])
    A = np.zeros((theta,theta))
    for i in range(theta):
        v2,v1 = V[i,:]
        for j in range(theta):
            Dp,z1 = Dpz[j,:]
            Tf1 = dma1.Transfer_function_Tf(Dp,v1,z1)
            Tl1 = dma1.Penetration_efficiency_Tl(Dp)
            A1 = Tl1 * Tf1
            Tl2 = dma2.Penetration_efficiency_Tl(Dp)
            for zn in z_n_vec:
                Tc2 = dma2.Charging_efficiency_Tc(Dp,zn)
                Tf2 = dma2.Transfer_function_Tf(Dp,v2,zn)
                A2 = Tl2 * Tc2 * Tf2
                A[i,j] += A1 * A2
    return A
def R1p_simplified(v2,
                   Dp2,
                   R,
                   dma2):
    theta = len(Dp2)
    R1p = np.zeros(theta)
    Tl2 = 1. #dma2.Penetration_efficiency_Tl(Dp)
    for i in range(theta):
        Tc2 = dma2.Charging_efficiency_Tc(Dp2[i],1)
        Tf2 = 1. #dma2.Transfer_function_Tf(Dp[i],v2,1)
        A2 = Tl2 * Tc2 * Tf2
        R1p[i] = R[i]/A2
    return R1p
def DMA1_conv_matrix_simplified(zDMA1,
                                V1,
                                Dp2,
                                dma1):
    theta = len(zDMA1)
    A1 = np.zeros((theta,theta))
    for i in range(theta):
        v1 = V1[i]
        for j in range(theta):
            z1 = zDMA1[j]
            Dp = Dp2[j]
            Tf1 = dma1.Transfer_function_Tf(Dp,v1,z1)
            Tl1 = dma1.Penetration_efficiency_Tl(Dp)
            A1[i,j] = Tl1 * Tf1
    return A1
def Deconvolute_singleDMA_charges(dma,
                                  z_vec,
                                  R0,
                                  Dp0,
                                  Voltage0,
                                  smooth_n=10,
                                  alpha = 1e-01,
                                  z_max = 40,
                                  n_z = 100,
                                  cubic_spline=True,
                                  check = False,
                                  with_z_new=True,
                                  title=""):
    #z_new = np.linspace(1,z_max+1,n_z)
    if(with_z_new):
        z_new = np.arange(1,z_max+1)
    else:
        z_new = z_vec.copy()
    if(cubic_spline):
        cs_R = CubicSpline(z_vec, R0)
        cs_v = CubicSpline(z_vec, Voltage0)
        cs_D = CubicSpline(z_vec, Dp0)
        R = cs_R(z_new)
        R[R<0] = 0.
        Voltage = cs_v(z_new)
        Dp = cs_D(z_new)
    else:
        R = np.interp(z_new,z_vec,R0)
        R[R<0] = 0.
        Voltage = np.interp(z_new,z_vec,Voltage0)
        Dp = np.interp(z_new,z_vec,Dp0)
        
    A = DMA1_conv_matrix_simplified(z_new,Voltage,Dp,dma)
    #A = dma.Convolution_matrix(Dp, Voltage)
    R_nr = Noise_reduction(R,5,2,"interp")
    
    N_alpha = Deconvolution(R_nr, A, alpha) #dma.Deconvolution(R_nr, A, alpha)
    R_conv = Convolution(A, N_alpha) #dma.Convolution(A, N_alpha)
    
    if (check==True):
        fig, ax = plt.subplots(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(z_vec, R0,"o b", label="Read DMA (original)")
        plt.plot(z_new, R,". r", label="Read DMA (cubic spline)")
        plt.plot(z_new, R_conv,"-g", label="R_conv")
        #dNdDp = Convert_dN_dDp(Dp,N_alpha)
        plt.plot(z_new, N_alpha,"--c", label="N_alpha")
        plt.ylabel("Particle size dist., $dR/dD_p$ (cm⁻³)", fontsize=20)
        plt.xlabel("Particle charges, z (-)", fontsize=20)
        ax.tick_params(direction='in', length=6, width=1, colors='k',
               grid_color='k', grid_alpha=0.5)
        ax.tick_params(axis='x', which='minor', direction='in')#,bottom=False)
        ax.tick_params(axis='y', which='minor', direction='in')#,bottom=False)
        plt.rc('xtick', labelsize=16); plt.rc('ytick', labelsize=16)
        plt.legend(fontsize=20); plt.grid()
        plt.title(title, fontsize=20)
        plt.show()
    R_conv_error = np.sqrt(np.power(R-R_conv,2))/R
    N_alpha = np.interp(z_vec,z_new,N_alpha)
    R_conv = np.interp(z_vec,z_new,R_conv)
    R_conv_error = np.interp(z_vec,z_new,R_conv_error)
    return N_alpha,R_conv,R_conv_error

##---------------------------------------------------##
## DMA calculations
##---------------------------------------------------##
def Find_Dp_givenV(data,dma):
    V = data.V2
    dma.Update_voltage(V)
    Zp = dma.Zp
    Dp = DMA_APS_inversion_tools.Search_Dp_givenZp(V, Zp)
    return Dp
def Find_DMA_central_Zp(data,dma_number,dma):
    if(dma_number == 1):
        voltage = data.V1
    else:
        voltage = data.V2
    dma.Update_voltage(voltage)
    return dma.Zp
def Search_z_givenZpDp(Z_pi, Dp, q_max=1000):
    T_g=300
    f = Aerosol_tools.friction(Dp,T_g)
    error = 1e+06
    z = 1
    for q in range(1,q_max+1):
        Zpi = float(q)*DMA_APS_inversion_tools.Aerosol_tools.q_e/f
        diff = np.abs(Zpi-Z_pi)/Z_pi
        if(diff < error):
            error = diff
            z = q
    return z,error
def Find_charge_givenDp(data):
    Dp = data.Dp2 * 1e-09
    Zp = data.Zp1
    z = Search_z_givenZpDp(Zp,Dp)
    return z
def Find_Dp_charges(data):
    V1s = np.unique(data.V1)
    V2s = np.unique(data.V2)
    
    # Vector of diameters
    Dp2 = np.unique(data[data.V1 == V1s[0]].Dp2.values)
    Dp2 *= 1e-09
    
    # Vector of charges (omit the largest ones)
    charges = np.arange(1,2*len(V1s)+1,2)
    
    print("len(V1), len(V2), len(charges), len(R)",len(V1s)," ",len(V2s)," ",len(charges)," ",len(data.R))
    print("charges: ",charges)
    print("Dp: ",np.round(Dp2 * 1e+09,2))
    
    t = len(charges) * len(Dp2)
    V = np.zeros((t,2))
    Dpz = np.zeros((t,2))
    k=0
    for i in range(len(Dp2)):
        for j in range(len(charges)):
            V[k,:] = [V2s[i], V1s[j]]
            Dpz[k,:] = [Dp2[i], charges[j]]
            k += 1
    return V,Dpz

##---------------------------------------------------##
## Noise reduction functions
##---------------------------------------------------##
def Noise_reduction(y,
                    window_size,
                    poly_order,
                    mod):
    yhat = savgol_filter(y,window_size,poly_order,mode=mod)
    return yhat
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
def Spline_eval(xp, x,y):
    from scipy.interpolate import CubicSpline
    f = CubicSpline(x, y, bc_type='natural')
    yp = f(xp)
    return yp
def Spline(x,y,points=100):
    x_new = np.linspace(np.min(x), np.max(x), points)
    y_new = Spline_eval(x_new, x,y)
    y_new[y_new<0] = 0.
    return x_new,y_new
def Spline_charge(x,y,max_charge=40):
    x_new = np.arange(1,max_charge+1)
    y_new = Spline_eval(x_new, x,y)
    y_new[y_new<0] = 0.
    return x_new,y_new


##---------------------------------------------------##
## Convert distributions
##---------------------------------------------------##
def Get_charge_fraction(z, N, zmax=300):
    z_interp = np.arange(1,zmax,1)
    N_sum = np.sum(np.interp(z_interp, z, N))
    f = N/N_sum
    return f
def Convert_dNdLogDp(Dp,N):
    dLog_DP = np.zeros_like(Dp)
    for i in range(1,len(Dp)):
        dLog_DP[i] = np.log10(Dp[i]/Dp[i-1])
    dLog_DP[0] = dLog_DP[1]
    dNdLogDp = N/dLog_DP
    return dNdLogDp
def Convert_dN_dDp(Dp,N,m_2_cm=1e+02):
    delta_DP = np.zeros_like(Dp)
    for i in range(1,len(Dp)):
        delta_DP[i] = (Dp[i]-Dp[i-1]) * m_2_cm # from m to cm
    delta_DP[0] = delta_DP[1]
    dN_alpha_dDp = N/delta_DP
    return dN_alpha_dDp
def Convert_dNdLogD_2_dNdD(Dp,dNdLogD):
    dLog_DP = np.zeros_like(Dp)
    for i in range(1,len(Dp)):
        dLog_DP[i] = np.log10(Dp[i]/Dp[i-1])
    dLog_DP[0] = dLog_DP[1]
    N = dNdLogD * dLog_DP
    dNdD = Convert_dN_dDp(Dp,N)
    return dNdD
def Convert_dNdLogD_2_dN(Dp,dNdLogD):
    dLog_DP = np.zeros_like(Dp)
    for i in range(1,len(Dp)):
        dLog_DP[i] = np.log10(Dp[i]/Dp[i-1])
    dLog_DP[0] = dLog_DP[1]
    N = dNdLogD * dLog_DP
    return N
def Determine_f_tot(V,Dpz,Dp0, dN_alpha_dDp0):
    f_tot = 0
    theta = len(V[:,0])
    for j in range(theta):
        Dp,z1 = Dpz[j,:]
        f_Dp = Spline_eval(Dp, Dp0, dN_alpha_dDp0)
        f_tot += f_Dp
    return f_tot
def Convert_dNdLogDp_PANDAS(measurements_data):
    ESPV = np.unique(measurements_data['ESP_V (kV)'])
    measurements_data["dNdLogDp"] = 0

    for espV in ESPV:
        data = measurements_data[measurements_data['ESP_V (kV)'] == espV]
        dNdLogDp = Convert_dNdLogDp(data["Dp"].values,data["R"])
        measurements_data = Set_column_ESPdata(measurements_data,
                                               dNdLogDp.index.values,
                                               dNdLogDp.values,
                                               "dNdLogDp")
    return measurements_data

##---------------------------------------------------##
## Pandas dtaframes management
##---------------------------------------------------##
def Set_column_ESPdata(df, idx, data, c_name):
    k=0
    for i in idx:
        df[c_name].iloc[i] = data[k]
        k += 1
    return df

##---------------------------------------------------##
## Plot results
##---------------------------------------------------##
def Plot_ESPV(espV,V2s, measurements_data_avg,
              line=False,
              column='R',
              row = "V1",
              spline=False,
              D_legend=False,
              LogYscale=True,
              xmax = None,
              export=False,
              sort_col="V2",
              skip=1,
              Dmax=2,
              prefix=""):
    x_charges = (row == "zDMA1" or row == "z1_raw" or row == "z1")
    data = measurements_data_avg[measurements_data_avg['ESP_V (kV)'] == espV]
    fig, ax = plt.subplots(num=None, figsize=(11, 7), dpi=80, facecolor='w', edgecolor='k')
    if(Dmax<2):
        V2s = V2s[V2s < Dmax]
    for i in range(len(V2s)): # sel: #
        V2 = V2s[i]
        sel_md = data[data[sort_col]==V2]
        if((len(sel_md) > 1) & (i%skip == 0)):
            R = sel_md[column]
            if (D_legend):
                legend = "$D_{p}=$ "+str(round(sel_md.Dp2.values[0],1))+" nm"
            else:
                legend = "$V_2=$ "+str(V2/1000)+" kV"
            if(row == "V1"):
                x = V1
            else:
                x = sel_md[row]
            if(line):
                if(spline):
                    inds = np.argsort(x.values)
                    R_sort = R.values[inds]
                    x_sort = x.values[inds]
                    xs,Rs = Spline(x_sort, R_sort)
                    plt.plot(xs, Rs, ".-", linewidth=1, label=legend)
                else:
                    plt.plot(x, R, ".-", linewidth=1, label=legend)
            else:
                plt.plot(x, R, "o", label=legend)
            del sel_md,R
    
    if(column == "R"):
        plt.ylabel("Particle size dist., $dR$ (cm⁻³)", fontsize=20)
    elif(column == "z1_rough" or column == "z1"):
        plt.ylabel("Elementary charges (-)", fontsize=20)
    elif((column == "N1p_nor") or (column == "f1")):
        plt.ylabel("Charge fraction, f (-)", fontsize=20)
    else:
        plt.ylabel("Particle size dist., $dN$ (cm⁻³)", fontsize=20)
    
    if(row == "Zp"):
        plt.xlabel("Electrical mobility $Z_p$ (m$^2$/(V s))", fontsize=20)
        plt.xscale("log");
    elif(row == "Dp"):
        plt.xlabel("Particle diameter $D_p$ ($\mu$m))", fontsize=20)
    elif(x_charges):
        plt.xlabel("Elementary charges (-)", fontsize=20)
    else:
        plt.xlabel("DMA-1 voltage $V_1$ (V)", fontsize=20)
    plt.title("tDMA, ESP V="+str(espV)+" kV",  fontsize=20) 
    if(LogYscale):
        plt.yscale("log");
        plt.ylim([1,1e+04])
    ax.tick_params(direction='in', length=6, width=1, colors='k',
               grid_color='k', grid_alpha=0.5)
    ax.tick_params(axis='x', which='minor', direction='in')#,bottom=False)
    ax.tick_params(axis='y', which='minor', direction='in')#,bottom=False)
    plt.rc('xtick', labelsize=16); plt.rc('ytick', labelsize=16)
    plt.legend(fontsize=11);
    if (xmax != None):
        plt.xlim([0,xmax])
    #plt.grid(b=False, which='minor', color='k', linestyle='--')
    plt.grid(); plt.show()
    if(export):
        plt.savefig('Figures/'+prefix+'ESP_v'+str(espV)+'_data_'+row+'.png');
    return
    
def Plot_profiles(data2, espV, Dp_fit, export=True,zmax=300,xmax=40):
    fig = plt.figure(num=None, figsize=(11, 7), dpi=80, facecolor='w', edgecolor='k');
    fig.set_tight_layout(False)
    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18)
    
    z1_raw = data2['z1_raw'].values
    N_raw = data2['R'].values
    f_raw = Get_charge_fraction(z1_raw, N_raw, zmax=zmax)
    plt.plot(z1_raw, f_raw,".-k",linewidth=0.5,label="Raw approximation")
    
    z_vec = data2['z1'].values
    N_deconv = data2['N_deconv'].values
    f1 = Get_charge_fraction(z_vec, N_deconv, zmax=zmax)
    plt.plot(z_vec, f1,"o r",label="Deconvolution")
    xs,Rs = Spline(z_vec, f1)
    plt.plot(xs,Rs,"-r",linewidth=2)
    
    plt.title('Charge profile, ESP V='+str(espV)+' kV, $D_p=$'+str(round(Dp_fit*1e+09,1))+' nm', fontsize=20)
    plt.ylabel('Charge fraction, f$_1$ (-)', fontsize=18)
    plt.xlabel('Elementary charges, z (-)', fontsize=18)
    plt.xlim([0,xmax])
    plt.legend(fontsize=15); plt.show()
    if(export):
        plt.savefig('Figures/'+"/ESP_V"+str(espV)+"kV_V2_"+str(V2_vec)+"_Dp"+str(round(Dp_fit,1))+"_JoseMoran.png")
    return