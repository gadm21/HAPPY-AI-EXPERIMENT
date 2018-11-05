def getRectangle(box,width,height):
    (ymin, xmin, ymax, xmax) = (
        int(box[0] * height),
        int(box[1] * width),
        int(box[2] * height),
        int(box[3] * width),
    )
    return (xmin,ymin,xmax,ymax)

def validBoundingBox(rect,score,label,selectedClass):
    (xmin,ymin,xmax,ymax) = rect
    if score < 0.7:
        return False
    if label not in selectedClass:
        return False
    ## context depends - if the object does not so big or small, just filter it
    # if ymax - ymin > 200 or xmax - xmin > 200:
    #     return False
    return True