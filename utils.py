def ft_sum(num_list):
    res = 0
    for num in num_list:
        res += num
    return res

def ft_sqrt(number):
    if number > 0:
        prec = 1
        i = 0
        while (prec >= 0.0001):
            while i * i <= number:
                if ft_abs(i * i - number) < 0.01:
                    return(i)
                i += prec
            i = 0
            prec /= 10
        return (0)

def ft_abs(number):
    if number > 0:
        return(number)
    else:
        return(-number)

def ft_power(l_member, r_member):
    i = r_member
    res = l_member
    if r_member != 0:
        if r_member % 1 != 0:
            res = l_member ** r_member
        elif r_member > 0:
            i = r_member - 1
        elif r_member < 0:
            i = r_member + 1
        while i and i % 1 == 0:
            if r_member > 0:
                i -= 1
                res *= float(l_member)
            elif r_member < 0:
                i += 1
                res *= float(l_member)
        if r_member < 0 and i % 1 == 0:
            res = 1 / res
    else:
        res = 1
    return (res)