import cv2


def main():
    img = cv2.imread("db/CUB_200_2011/images/009.Brewer_Blackbird/Brewer_Blackbird_0064_2290.jpg")
    cv2.imshow("Before", img)
    cv2.waitKey(0)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decoded_img = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    cv2.imshow("After", decoded_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
