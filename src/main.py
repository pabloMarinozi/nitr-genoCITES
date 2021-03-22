import argparse
from CustomGoogleMapPlotter import CustomGoogleMapPlotter
from index_calculation import get_index_image
from heatmap_generation import generate_heatmap,plt,generate_heatmap_video
from GMM_segmentation import os,cv2,np,segment_leaf_GMM
from gpsFileReading import asignGPS2Frames

# error message when image could not be read
IMAGE_NOT_READ = 'IMAGE_NOT_READ'

# error message when image is not colored while it should be
NOT_COLOR_IMAGE = 'NOT_COLOR_IMAGE'

if __name__ == '__main__':
    # handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--destination',
                        help='Destination directory for output image. '
                             'If not specified destination directory will be input image directory')
    parser.add_argument('-l', '--leaf_model_path',
                        help='a path to the binary file containing the leaf detection model')
    parser.add_argument('-b', '--background_model_path',
                        help='a path to the binary file containing the background detection model')
    parser.add_argument('-o', '--with_original', action='store_true',
                        help='Segmented output will be appended horizontally to the original image')
    parser.add_argument('-i', '--index', choices=['r', 'b', 'g', 'rb','rg','bg'],
                        help='Index used to generate heatmap.',
                        default='r')
    parser.add_argument('image_source', help='A path of image filename or folder containing images')
    parser.add_argument('gps_source', help='A path of xml filename containing the coordinates for each second')
    parser.add_argument('frame_rate', help='The number of frames corresponding to each second of the video')
    
    # set up command line arguments conveniently
    args = parser.parse_args()
    api_key = "AIzaSyA7hF-pmJqAf26TmG90EaT2tG7wJuBb5rA"
    if args.destination:
        if not os.path.isdir(args.destination):
            print(args.destination, ': is not a directory')
            exit()

    # set up files to be segmented and destination place for segmented output
    if os.path.isdir(args.image_source):
        files = [entry for entry in os.listdir(args.image_source)
                 if os.path.isfile(os.path.join(args.image_source, entry))]
        files.sort()
        base_folder = args.image_source

        # set up destination folder for segmented output
        if args.destination:
            destination = args.destination
        else:
            if args.image_source.endswith(os.path.sep):
                args.image_source = args.image_source[:-1]
            destination = args.image_source + '_markers'
            os.makedirs(destination, exist_ok=True)
    else:
        folder, file = os.path.split(args.image_source)
        files = [file]
        base_folder = folder

        # set up destination folder for segmented output
        if args.destination:
            destination = args.destination
        else:
            destination = folder
            
   
    #process each frame
    heatmap_files = []
    agregateds = []
    for n,file in enumerate(files):
        try:
            # read image and segment leaf
            original, output_image = \
                segment_leaf_GMM(os.path.join(base_folder, file),
                                 args.leaf_model_path, args.background_model_path)

        except ValueError as err:
            if str(err) == IMAGE_NOT_READ:
                print('Error: Could not read image file: ', file)
            elif str(err) == NOT_COLOR_IMAGE:
                print('Error: Not color image file: ', file)
            else:
                raise
        # if no error when segmenting write segmented output
        else:
            # handle destination folder and fileaname
            filename, ext = os.path.splitext(file)
            if args.with_original:
                new_filename = filename + '_marked_merged' + ext
            else:
                new_filename = filename + '_marked' + ext
            new_filename = os.path.join(destination, new_filename)
            
            #calculate index for each pixel
            index_image = get_index_image(output_image, args.index)
            
            #generate heatmap from index_image
            heatmap_filename = filename + '_heatmap' + ext
            heatmap_filename = os.path.join(destination, heatmap_filename)
            masked_index_image = np.ma.masked_where(index_image==0, index_image)
            generate_heatmap(masked_index_image, heatmap_filename)
            heatmap_files.append(heatmap_filename)
            
            #generate an index for the entire frame
            agregated_index = np.sum(masked_index_image)/masked_index_image.count()
            if n>=int(args.frame_rate): agregateds.append(agregated_index) 

            # write the output
            if args.with_original:
                cv2.imwrite(new_filename, np.hstack((original,output_image)))
            else:
                cv2.imwrite(new_filename, output_image)
            print('Marker generated for image file: ', file)
       
    #create heatmap video from heatmap image files
    img_list = []
    for filename in heatmap_files:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_list.append(img)
    video_filename = os.path.join(destination,'heatmap.avi')
    generate_heatmap_video(img_list, size, video_filename)
    
    #get gps coordinates for each frame
    x_array, y_array = asignGPS2Frames(args.gps_source,len(files),int(args.frame_rate))
    
    print("x, ",type(x_array),x_array)
    print("y, ",type(y_array),y_array)
    
    #create nitrogen map 
    agregateds = np.array(agregateds)
    gmap = CustomGoogleMapPlotter(np.mean(x_array), np.mean(y_array), 12,
                                  "AIzaSyA7hF-pmJqAf26TmG90EaT2tG7wJuBb5rA", 'roadmap')
    gmap.apikey = "AIzaSyA7hF-pmJqAf26TmG90EaT2tG7wJuBb5rA"
    gmap.color_scatter(x_array.tolist(), y_array.tolist(), agregateds, colormap=plt.cm.RdYlGn_r,size=2)
    map_path = "mymap.html"
    map_path = os.path.join(destination, map_path)
    gmap.draw(map_path)
    print('Nitrogen map generated at: ', map_path)
