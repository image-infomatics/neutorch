def fix_volume(vol, path, offset=37):
    if 'A+' in path:
        return fix_A(vol, offset=offset)
    elif 'B+' in path:
        return fix_B(vol, offset=offset)
    elif 'C+' in path:
        return fix_C(vol, offset=offset)
    else:
        print(f'WARN no fix method found for {path}')
        return vol


def fix_A(vol, offset=37):
    # problem slices
    # 0 33 51 79 80 108 109 111
    o = offset
    vol[..., o+0, :, :] = vol[..., o+1, :, :]
    vol[..., o+33, :, :] = vol[..., o+34, :, :]
    vol[..., o+51, :, :] = vol[..., o+52, :, :]
    vol[..., o+79, :, :] = vol[..., o+78, :, :]
    vol[..., o+80, :, :] = vol[..., o+81, :, :]
    vol[..., o+108, :, :] = vol[..., o+107, :, :]
    vol[..., o+109, :, :] = vol[..., o+110, :, :]
    vol[..., o+111, :, :] = vol[..., o+112, :, :]
    return vol


def fix_B(vol, offset=37):
    # problem slices
    # 15 16 44 45 ~74 77
    o = offset
    vol[..., o+15, :, :] = vol[..., o+14, :, :]
    vol[..., o+16, :, :] = vol[..., o+17, :, :]
    vol[..., o+44, :, :] = vol[..., o+43, :, :]
    vol[..., o+45, :, :] = vol[..., o+46, :, :]
    vol[..., o+74, :, :] = vol[..., o+73, :, :]
    vol[..., o+77, :, :] = vol[..., o+78, :, :]
    return vol


def fix_C(vol, offset=37):
    # problem slices
    # 14 74 86
    o = offset
    vol[..., o+14, :, :] = vol[..., o+15, :, :]
    vol[..., o+74, :, :] = vol[..., o+73, :, :]
    vol[..., o+86, :, :] = vol[..., o+87, :, :]
    return vol
