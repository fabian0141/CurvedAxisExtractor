def splitContours(con, img):
    miniCons = []
    i = 0
    leng = len(con)
    firstIdx = None
    count = 0

    while True:
        max = 0

        #i10 = (i + 10) % leng 
        if firstIdx != None:
            outer = np.min([firstIdx - i, 20])
        else: 
            outer = 20

        i20 = (i + outer) % leng 
        idx = (i+10) % leng



        for j in range(outer):
            ij = (i+j) % leng
            dist = distancePointToLine(con[i], con[i20], con[ij])
            if dist > 1 and dist > max:
                #cv.circle(img, con[ij].toIntArr(), 0, (0, 0, 0), 1)
                idx = ij
                max = dist

        if outer < 20 and max == 0:
            break

        if idx >= i:
            count += idx - i
        else:
            count += idx + leng - i

        if firstIdx != None and count >= firstIdx:
            break



        miniCons.append(idx)
        #cv.circle(img, con[idx].toIntArr(), 1, (255, 0, 0), 1)
        #cv.line(img, con[i].toIntArr(), con[idx].toIntArr(), (150,150,150), 1)



        i = idx


        if firstIdx == None:
            firstIdx = leng + i
           #cv.circle(img, con[idx].toIntArr(), 1, (255, 0, 0), 1)
           #cv.line(img, con[idx].toIntArr(), con[i+10].toIntArr(), (150,150,150), 1)

    # TODO: handle end by finding corners inside

    return miniCons