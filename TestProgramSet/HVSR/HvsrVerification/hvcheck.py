from numpy import *


def hvsrcheck(indict):
    print("-----------------------------------------------------------------------")
    print("File Name\t\t: %s" % indict['filename'])
    print("-----------------------------------------------------------------------")
    print("CRITERIA FOR A RELIABLE H/V CURVE")
    print("-----------------------------------------------------------------------")
    # Check for reliable H/C curve
    # (1) f0 > 10/window length
    if indict['F0'] > (10/indict['winlength']): # the output is True/False
        print("RELIABLE 1 - OK\t\t:f0 > 10/lw\t= %f > %f" % (indict['F0'], 10/indict['winlength']))
    else:
        print("RELIABLE 1 - NO\t\t:f0 < 10/lw\t= %f < %f" % (indict['F0'], 10/indict['winlength']))
    # (2) nc(f0) > 200
    nc = indict['winlength']*indict['window']*indict['F0']
    if nc > 200:  # the output is True/False
        print("RELIABLE 2 - OK\t\t:nc(f0) > 200\t= %f > 200" % nc)
    else:
        print("RELIABLE 2 - NO\t\t:nc(f0) < 200\t= %f < 200" % nc)
    # (3) std_H/V(f) < 2 or std_H/V(f) < 3
    if indict['F0'] > 0.5:
        idfr1 = where(indict['frhv'] > 0.5*indict['F0'])[0]
        idfr2 = where(indict['frhv'] < 2*indict['F0'])[0]
        r31 = all(indict['stdhv'][idfr1] < 2)
        r32 = all(indict['stdhv'][idfr2] < 2)
        if (r31 == True) and (r32 == True):
            print("RELIABLE 3 - OK\t\t:Std H/V(f) < 2\t= %f > 0.5Hz" % indict['F0'])
        else:
            print("RELIABLE 3 - NO\t\t:Std H/V(f) > 2\t= %f > 0.5Hz" % indict['F0'])
    elif indict['F0'] < 0.5:
        idfr1 = where(indict['frhv'] > 0.5*indict['F0'])[0]
        idfr2 = where(indict['frhv'] < 2*indict['F0'])[0]
        r31 = all(indict['stdhv'][idfr1] < 3)
        r32 = all(indict['stdhv'][idfr2] < 3)
        if (r31 == True) and (r32 == True):
            print("RELIABLE 3 - OK\t:Std H/V(f) < 3\t= %f > 0.5Hz" % indict['F0'])
        else:
            print("RELIABLE 3 - NO\t:Std H/V(f) > 3\t= %f > 0.5Hz" % indict['F0'])

    print("-----------------------------------------------------------------------")
    print("CLEAR PEAK FOR H/V CURVE")
    print("-----------------------------------------------------------------------")
    # Check for Clear Peak of H/V
    # (1) H/V(f-) < A0/2 for f0/4 to f0
    clear_count = []
    idfr1 = where(indict['frhv']>indict['F0']/4)[0]
    idfr2 = where(indict['frhv']<indict['F0'])[0]
    c1 = (indict['hvsr'][idfr1] < indict['A0']/2)
    c2 = (indict['hvsr'][idfr2] < indict['A0']/2)
    c1 = True in c1
    c2 = True in c2
    if (c1 == True) and (c2 == True):
        cc = 1
        print("CLEAR PEAK 1 - OK\t:\u2203 f- \u2208 [f0/4,f0] \u2223 A(f-) < A0/2")
    else:
        cc = 0
        print("CLEAR PEAK 1 - OK\t:\u2203 f- \u2208 [f0/4,f0] \u2223 A(f-) > A0/2")
    clear_count.append(cc)

    # (2) H/V(f+) < A0/2 for f0 to 4f0
    idfr3 = where(indict['frhv'] > indict['F0'])[0]
    idfr4 = where(indict['frhv'] < 4*indict['F0'])[0]
    c3 = (indict['hvsr'][idfr3] < indict['A0']/2)
    c4 = (indict['hvsr'][idfr4] < indict['A0']/2)
    c3 = True in c3
    c4 = True in c4
    if (c3 == True) and (c4 == True):
        cc = 1
        print("CLEAR PEAK 2 - OK\t:\u2203 f+ \u2208 [f0,4f0] \u2223 A(f+) < A0/2")
    else:
        cc = 0
        print("CLEAR PEAK 2 - OK\t:\u2203 f+ \u2208 [f0,4f0] \u2223 A(f+) > A0/2")
    clear_count.append(cc)

    # (3) A0 > 2
    if indict['A0'] > 2:
        cc = 1
        print("CLEAR PEAK 3 - OK\t:A0 > 2 |", indict['A0'], "> 2")
    else:
        cc = 0
        print("CLEAR PEAK 3 - NO\t:A0 < 2 |", indict['A0'], "< 2")
    clear_count.append(cc)

    # (4) fpeak[A(f) +/- stdA(f) = f0 +/- 5%]
    F0min = indict['F0']-(indict['F0']*(5/100))
    F0max = indict['F0']+(indict['F0']*(5/100))
    idminstdhv = argmax(indict['minstdhv'])
    idmaxstdhv = argmax(indict['maxstdhv'])
    if (indict['frhv'][idminstdhv] > F0min) and (indict['frhv'][idminstdhv] < F0max):
        if (indict['frhv'][idmaxstdhv] > F0min) and (indict['frhv'][idmaxstdhv] < F0max):
            cc = 1
            print("CLEAR PEAK 4 - OK\t:Fpeak [A(f) \u00B1 stdA(f) = f0 \u00B1 5%]")
        else:
            cc = 0
            print("CLEAR PEAK 4 - NO\t:Fpeak [A(f) \u00B1 stdA(f)\u2260f0 \u00B1 5%]")
    clear_count.append(cc)

    # (5) stdF < epsilon(f0)
    if indict['F0'] < 0.2:
        epsf0 = 0.25*indict['F0']
    elif (indict['F0'] > 0.2) and (indict['F0'] < 0.5):
        epsf0 = 0.2*indict['F0']
    elif (indict['F0'] > 0.5) and (indict['F0'] < 1.0):
        epsf0 = 0.15*indict['F0']
    elif (indict['F0'] > 1.0) and (indict['F0'] < 2.0):
        epsf0 = 0.10*indict['F0']
    elif indict['F0'] > 2.0:
        epsf0 = 0.05*indict['F0']

    if indict['stdf0'] < epsf0:
        cc = 1
        print("CLEAR PEAK 5 - OK\t:\u03C3f < \u03B5(F0) |",indict['stdf0'], "<", epsf0)
    else:
        cc = 0
        print("CLEAR PEAK 5 - NO\t:\u03C3f > \u03B5(F0) |",indict['stdf0'], ">", epsf0)
    clear_count.append(cc)

    # (6) stdA(f0) < theta(f0)
    if indict['F0'] < 0.2:
        thetaf0 = 0.3
    elif (indict['F0'] > 0.2) and (indict['F0'] < 0.5):
        thetaf0 = 2.5
    elif (indict['F0'] > 0.5) and (indict['F0'] < 1.0):
        thetaf0 = 2.0
    elif (indict['F0'] > 1.0) and (indict['F0'] < 2.0):
        thetaf0 = 1.78
    elif indict['F0'] > 2.0:
        thetaf0 = 1.58

    if indict['stdA'] < thetaf0:
        cc = 1
        print("CLEAR PEAK 6 - OK\t:\u03C3A(F0) < \u03B8 (F0)")
    else:
        cc = 0
        print("CLEAR PEAK 6 - OK\t:\u03C3A(F0) > \u03B8 (F0)")
    clear_count.append(cc)

    print("\nCLEAR PEAK SUMMARY\t: %d out of 6 criteria fulfilled" % int(sum(clear_count)))
    if int(sum(clear_count)) >= 5:
        print("CLEAR PEAK SUMMARY\t: H/V IS CLEAR PEAK")
    else:
        print("CLEAR PEAK SUMMARY\t: H/V IS NOT CLEAR PEAK [at least 5 out of 6 criteria fulfilled]")
