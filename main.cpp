// [includes]
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#include <ctime>
#include <vector>
#include <deque>
#include <set>
#include <igraph.h>


//---------------------------------------------------------------------------//
//[namespace]
using namespace cv;
using namespace std;
//---------------------------------------------------------------------------//
#define sep "/"
//---------------------------------------------------------------------------//
struct Couple {
    int k0;
    int k1;
    
    // default parameterized constructor
    Couple(int k0 = 0, int k1 = 0) : k0(k0), k1(k1) {
    }
    
    // addition operator
    Couple operator+(const Couple& a) const {
        return Couple(this->k0 + a.k0, this->k1 + a.k1);
    }
    
    bool operator<(Couple a) const{
        if(a.k0 < k0)return true;
        if(a.k0 == k0 && a.k1 < k1)return true;
        else return false;
    }
};
// doubling operator
Couple dblCouple(const Couple& a) {
    return Couple(2*a.k0, 2*a.k1);
}
//---------------------------------------------------------------------------//
struct Triple{
    int k0;
    int k1;
    int k2;
    
    // default parameterized constructor
    Triple(int k0 = 0, int k1 = 0, int k2 = 0) : k0(k0), k1(k1), k2(k2) {
    }
    
    bool operator<(Triple a) const{
        if(a.k0 < k0)return true;
        if(a.k0 == k0 && a.k1 < k1)return true;
        else return false;
    }

};
//---------------------------------------------------------------------------//
struct DTriple{
    double k0;
    double k1;
    int k2;
    
    // default parameterized constructor
    DTriple(double k0 = 0, double k1 = 0, int k2 = 0) : k0(k0), k1(k1), k2(k2) {
    }
    
    bool operator<(DTriple a) const{
        if(a.k0 < k0)return true;
        if(a.k0 == k0 && a.k1 < k1)return true;
        else return false;
    }
};
//---------------------------------------------------------------------------//
struct Adaptive_Grid {
    std::vector<Triple> Edges;
    std::vector<DTriple> pos;
    string dir_conv;
    double Disvalue;
};
//---------------------------------------------------------------------------//
struct Pos {
    double x;
    double y;
    int Depth;
    float Threshold;
    
    // default parameterized constructor
    Pos(double x, double y, int Depth, float Threshold) : x(x), y(y), Depth(Depth), Threshold(Threshold) {
    }
};
//---------------------------------------------------------------------------//
struct Graph{
    std::vector<Triple> Edges;
    std::vector<int> Keys;
};
//---------------------------------------------------------------------------//
struct Data_Label{
    string label;
    std::vector<int> data;
};
//---------------------------------------------------------------------------//
struct Labels{
};
//---------------------------------------------------------------------------//
struct Pos2D3D{
    std::vector<Triple> pos2D;
    std::vector<Triple> pos3D;
};
typedef map< int, std::vector<Couple> > nested_dict2;
typedef map< Couple, std::vector<Pos> > mapelement;
typedef map< int,  mapelement> nested_dict;
typedef vector< vector<double> > matrix;
//---------------------------------------------------------------------------//
void image_graph(int imWidth,int imHeight,string crd, Mat image, int smin, int thresholding_m);
void image_graph_run();
void image_graph_calc(string crd,string dir_input,string file_input);
Adaptive_Grid image_graph_AMR_2D_Adaptive_grid(int imWidth,int imHeight, string crd, Mat im, string dir_conv,string dir_Edges,string dir_QuadTree,string dir_output);
igraph_t grid_grid_all(Mat im, string file_path, Adaptive_Grid * AG, int dz);
void save0(string url,std::vector<Triple> Edges);
void save0(string url,std::vector<int> data);
void save0(string url,string label);
Data_Label graph_graph_all(Graph G,std::vector<DTriple> pos);
Labels load0(string url);
void write0(string url,Labels labels);
void plot0(string url, int imHeight, int imWidth, std::vector<DTriple> pos);
Labels load0(string url);
Graph load1(string url);
int number_of_nodes(Graph g);
Graph subgraph(Graph g, int n1, int n2);
Graph convert_node_labels_to_integers(Graph g);
Pos2D3D image_graph_help_grid_pos2D(int n, std::vector<DTriple> pos);
void save0(string url, std::vector<double> data);
//---------------------------------------------------------------------------//
void image_graph_run()
{
    string dir_input("/Users/vincUk/Desktop/image_graph/images/");
//    string file_name("Control1.tif");
    string file_name("Cytoskeletal-1.tif");
    string gridtype("rectangular");
    cout << "Starting for file \"" + file_name + "\"..." << endl;
    cout << "Grid type: " + gridtype + "." << endl;
    image_graph_calc(gridtype,dir_input,file_name);
    return;
}
//---------------------------------------------------------------------------//
void image_graph_calc(string crd,string dir_input,string file_input)
{
    string name("Adaptive_grid_");
    string dir_output = dir_input+"Output_"+name+file_input+sep;
    string subfolders[7] = {"data_posi","data_conv","data_grph","data_datn","data_prop","data_readable","plot_grid"};
    string file_path = dir_input+file_input;
    mkdir(dir_output.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    for (int i = 0; i < (sizeof(subfolders)/sizeof(*subfolders)); ++i)
    {
        string s1=dir_output+subfolders[i];
        mkdir(s1.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        //cout<<s1<<endl;
    }
    string s2=dir_output+"data_readable"+sep+"data_readable.txt";
    remove(s2.c_str());
    
    cout<<"image_graph_AMR"<<endl;
    Mat im = imread(file_path.c_str(), IMREAD_GRAYSCALE);
    
    int imHeight = im.rows;
    int imWidth = im.cols;
    int AMR = 1;///to be removed
    string dir_conv = dir_output+"data_conv"+sep+"data_conv";
    string dir_Edges = dir_output+"data_Edges"+sep+"data_Edges";
    string dir_QuadTree = dir_output+"data_QuadTree"+sep+"data_QuadTree";
    
    if(im.dims == 2)
    {
        printf("2D image_graph ..\n");
    }
    else
    {
        printf("3D image_graph ..\n"); // in next steps I will work on 3D version
    }
    Adaptive_Grid AdGrid = image_graph_AMR_2D_Adaptive_grid(imWidth, imHeight, crd, im, dir_conv, dir_Edges, dir_QuadTree, dir_output);
   
    cout<<"graph"<<endl;
    time_t temp1 = time(0);
    igraph_t graph = grid_grid_all(im, file_path, &AdGrid, 1);
    time_t temp2 = time(0);
//    save0(dir_output+sep+"data_grph"+sep+"data_grph.npy",graph.Edges);
    
    cout << "the time consumed to build the graph is " << difftime(temp2, temp1) << endl;
    cout<< "obs network" << endl;
    
    temp1 = time(0);
//    Data_Label data_label = graph_graph_all(graph,Edges_pos_dir_conv.pos);
    temp2 = time(0);
    cout<<"the time consumed to measure the graph's properties quantitatively is " << difftime(temp2, temp1) << endl;
//    save0(dir_output+sep+"data_datn"+sep+"data_datn.npy", data_label.data);
//    save0(dir_output+sep+"data_prop"+sep+"data_prop.npy", data_label.label);
    
//    Labels labels=load0(dir_output+sep+"data_prop"+sep+"data_prop.npy");
    
//    write0(dir_output+sep+"data_readable"+sep+"data_readable.txt",labels);
    // with open(dir_output"+sep+"data_readable"+sep+"data_readable.txt","a") as out:
    //     out.write('\t'.join([str(a) for a in numpy.hstack(labels)]))
    //     out.write('\n')
    //     datn=numpy.load(os.path.join(dir_output,'data_datn','data_datn.npy'))
    //     out.write('\t'.join([str(a) for a in numpy.hstack(datn)]))
    //     out.write('\n')
    
//    plot0(dir_output+sep+"data_grph"+sep+"data_grph.npy",imHeight,imWidth,Edges_pos_dir_conv.pos);
}
//---------------------------------------------------------------------------//
float threshold_otsu(Mat * im) {
    return 0;
}
float threshold(Mat * img, int k, int x1, int x2, int y1, int y2) {
    Mat imc = img->rowRange(y1, y2);
    imc = imc.colRange(x1, x2);
    
    if (k==1) {      //# THRESH_Otsu***
         return threshold_otsu(&imc);
    }
    else if (k==2) {   //# default Sauvola's thresholding method***
        Scalar m, std;
        meanStdDev(imc, m, std);
        float img_thresh = m[0];
        float img_std= std[0];
        return img_thresh * ( 1 + 0.2 * ( img_std / 128 - 1 ) );
    }
    else if (k==3) {   //# Niblack thresholding method using (mean)
        Scalar m, std;
        meanStdDev(imc, m, std);
        float img_thresh = m[0];
        float img_std= std[0];
        return img_thresh + 0.5 * img_std;
    }
    return -1;
}

//---------------------------------------------------------------------------//
float QudtreeThreshold (Mat * im, int k, vector<Pos> * posi, int Depth) {
    float B1_thresh = threshold(im, k, posi->operator[](0).x, posi->operator[](4).x, posi->operator[](0).y, posi->operator[](4).y);
    float B2_thresh = threshold(im, k, posi->operator[](1).x, posi->operator[](5).x, posi->operator[](1).y, posi->operator[](5).y);
    float B3_thresh = threshold(im, k, posi->operator[](3).x, posi->operator[](7).x, posi->operator[](3).y, posi->operator[](7).y);
    float B4_thresh = threshold(im, k, posi->operator[](4).x, posi->operator[](8).x, posi->operator[](4).y, posi->operator[](8).y);
    return min(min(B1_thresh, B2_thresh), min(B3_thresh, B4_thresh));
};
//---------------------------------------------------------------------------//
void FindPositions(vector<Pos> * posi, double x1, double y1, double dx, double dy, string crd, int Depth, float B_thresh1) {
    int n = 0;
    for (int j= 0; j < 3; j++) {
        for (int i = 0; i < 3; i++) {
            if (crd == "rectangular" ) posi->push_back(Pos((x1+dx*i), (y1+dy*j), Depth, B_thresh1));
            n++;
        }
    }
    printf("number of positions in Quads:%d\n", n);
}
//---------------------------------------------------------------------------//
int Divide_Decision(nested_dict * PS, vector<float> * Threshold, int Depth, int k, Couple cellCoords, double disvalue, float smin, int dmax, Mat * im, int imWidth, int imHeight, string crd, int D) {
    double cellWidth = double(imWidth) / pow(2,Depth);
    double cellHeight = double(imHeight) / pow(2,Depth);
    
    double dx = double(cellWidth/2);
    double dy = double(cellHeight/2);
    
    double x1 = (cellWidth*cellCoords.k0) + D; //#+0.1
    double y1 = (cellHeight*cellCoords.k1) + D;
    double x2 = (cellWidth*(cellCoords.k0 + 1)) + D;
    double y2 = (cellHeight*(cellCoords.k1 + 1)) + D;
    
    double mind = min(dx,dy);
    double minEdgeSize = min(cellWidth, cellHeight);
    
    if  (minEdgeSize >= smin) { // # comparing the boundaries with minimum size of the face
        float B_thresh1 = threshold(im, k, x1, x2, y1, y2);
        vector<Pos> posi;
        FindPositions(&posi, x1, y1, dx, dy, crd, Depth, B_thresh1);
        
        if (PS->find(Depth) == PS->end()) {
            mapelement mp;
            mp[cellCoords] = posi;
            PS->insert(pair<int, mapelement >(Depth, mp));
        }
        else
            PS->at(Depth)[cellCoords] = posi;
        
        if (mind >= smin) {
            float avgDthreshold = QudtreeThreshold(im, k, &posi, Depth);
            float AddedValue = Threshold->operator[](Depth) + avgDthreshold;
            Threshold->operator[](Depth) = AddedValue;
            printf("the sub-quads threshoding = %f \n", avgDthreshold);
            return 0;
        }
        else {
            if (int(mind) >= 2) {
                float avgDthreshold = QudtreeThreshold(im, k, &posi, Depth);
                float AddedValue = Threshold->operator[](Depth) + avgDthreshold;
                Threshold->operator[](Depth) = AddedValue;
                return 2;
            }
        }
    }
    else {
        cout << " This bin is smaller than than the Edge minimum size " << endl;
        return 1;
    }
    return -1;
}
//---------------------------------------------------------------------------//
void checkCell(nested_dict2 * QT, nested_dict * PS, vector<float> * Threshold, int Depth, int k, Couple cellCoords, float disvalue, float smin, Mat * im, int imWidth, int imHeight, string crd, int D, int dmax) {
    if (Depth > dmax+1) {
        cout << "the grid reaches the maximum depth " << Depth << endl;
        return;
    }
    //# Dividing the current depth for 4 parts :
    int minThreshold = Divide_Decision(PS, Threshold, Depth, k, cellCoords, disvalue, smin, dmax, im, imWidth, imHeight, crd, D);
    if (minThreshold  != 1) {
        if (QT->find(Depth) == QT->end()) {
            vector<Couple> v = {cellCoords};
            QT->insert(pair<int, vector<Couple> >(Depth,v));
        }
        else
            QT->at(Depth).push_back(cellCoords);
        if (minThreshold  != 2) {
            checkCell(QT, PS, Threshold, Depth + 1, k, dblCouple(cellCoords), disvalue, smin, im, imWidth, imHeight, crd, D, dmax);
            checkCell(QT, PS, Threshold, Depth + 1, k, dblCouple(cellCoords) + Couple(0,1), disvalue, smin, im, imWidth, imHeight, crd, D, dmax);
            checkCell(QT, PS, Threshold, Depth + 1, k, dblCouple(cellCoords) + Couple(1,0), disvalue, smin, im, imWidth, imHeight, crd, D, dmax);
            checkCell(QT, PS, Threshold, Depth + 1, k, dblCouple(cellCoords) + Couple(1,1), disvalue, smin, im, imWidth, imHeight, crd, D, dmax);
        }
    }
    
};
//---------------------------------------------------------------------------//
bool sortfunct (DTriple i, DTriple j) {
    if (i.k1 < j.k1) return true;
    if (i.k1 == j.k1 ) {
        if (i.k0 < j.k0) return true;
        else return false;
    }
    return false;
}
//---------------------------------------------------------------------------//
Triple most_common(vector<Triple> * inlist) {
    map<String, int> mp;
    int maxv = 0;
    Triple com = inlist->operator[](0);
    for (vector<Triple>::iterator it = inlist->begin(); it != inlist->end(); it++) {
        String str = to_string(it->k0)+to_string(it->k1);
        mp[str]++;
        if (mp[str] > maxv) {
                maxv = mp[str];
                com = *it;
        }
    }
    return com;
}
//---------------------------------------------------------------------------//
Adaptive_Grid Generate_Edges_Convs(long Depth, Couple cellCoords, Mat * im, double DisValue, int imWidth,int imHeight, int MinSize, vector<DTriple> * Posit, string dir_conv) {
    Adaptive_Grid ag;
    ag.Disvalue = DisValue;
    ag.dir_conv = dir_conv;
    
    double dx = float(imWidth) / pow(2, Depth);
    double dy = float(imHeight) / pow(2, Depth);

    // Generating the Edges
    vector<Triple> Edges;
    for (int key1 = 0; key1 < Posit->size(); key1++) {
        for (int key2 = 0; key2 < key1; key2++) {
            int Dx = abs(Posit->operator[](key1).k0 - Posit->operator[](key2).k0);
            int Dy = abs(Posit->operator[](key1).k1 - Posit->operator[](key2).k1);
            if (pow(float(Dx)/dx, 2) + pow(float(Dy)/dy, 2) < 1.1)
                Edges.push_back(Triple(key1, key2, 1));
        }
    }
    
    // Finding Edges' nodes in the main grid
    set<int> connectedNodes;
    Triple z1 = most_common(&Edges);
    set<int> visited, QV;
    QV.insert(z1.k0);
    while (QV.size() > 0) {
        int node = *QV.begin();
        QV.erase(QV.find(node));
        if (visited.find(node) == visited.end()) {
            for (int i=0; i< Edges.size(); i++) {
                if ((node == Edges[i].k0) || (node == Edges[i].k1)) {
                    if (connectedNodes.find(Edges[i].k0) == connectedNodes.end())
                        connectedNodes.insert(Edges[i].k0);
                    if (connectedNodes.find(Edges[i].k1) == connectedNodes.end())
                        connectedNodes.insert(Edges[i].k1);
                    if (QV.find(Edges[i].k0) == QV.end())
                        QV.insert(Edges[i].k0);
                    if (QV.find(Edges[i].k1) == QV.end())
                        QV.insert(Edges[i].k1);
                }
            }
            visited.insert(node);
        }
    }
    
    vector<DTriple> FPosit;
    
    for (int i = 0; i < Posit->size(); i++) {
        if (connectedNodes.find(i) != connectedNodes.end())
            FPosit.push_back(Posit->operator[](i));
    }
    
    ag.pos = FPosit;
    printf("# of Positions = %lu\n", FPosit.size());
    
    vector<Triple> FinalEdges;
    for (int key1 = 0; key1 < FPosit.size(); key1++) {
        for (int key2 = 0; key2 < key1; key2++) {
            int Dx = abs(FPosit[key1].k0 - FPosit[key2].k0);
            int Dy = abs(FPosit[key1].k1 - FPosit[key2].k1);
            if (pow(float(Dx)/dx, 2) + pow(float(Dy)/dy, 2) < 1.1)
                FinalEdges.push_back(Triple(key1, key2, 1));
        }
    }
    
    ag.Edges = FinalEdges;
    printf( "# of Edges is %lu \n", FinalEdges.size());
    
    return ag;
}
//---------------------------------------------------------------------------//
Adaptive_Grid image_graph_AMR_2D_Adaptive_grid(int imWidth,int imHeight, string crd, Mat im, string dir_conv, string dir_Edges, string dir_QuadTree, string dir_outpu)
{
    Adaptive_Grid ag;
    time_t starttime;
    time(&starttime);
    printf( "image width = %d, image height = %d\n", imWidth, imHeight);
    int k = 3;
    float smin = 3.2; //# thresholding works on 2px and more.
    if (smin < 2) {
        cout << " The minimum block size can not be less than 2 pixels" << endl;;
        return ag;
    }
    int D = int(ceil(log(float(min(imWidth, imHeight))/smin) + float(3/2))) + 1; //because depth start from 0 in Python
    int W = imWidth - (D*2);
    int H = imHeight - (D*2);
    printf( "imWidth of the grid = %d , imHeight of the grid = %d\n", W, H);
    int dmax = int(ceil(log(float(min(W, H))/smin) + float(3/2)));
    float disvalue;
    smin <= 16 ? disvalue= 0.5*smin : disvalue = 8; // set Guassian kernel
    printf("smin = %d, dmax = %d, Distance = %d\n", int(smin), dmax, D);
    printf("image width = %d, image height = %d\n", W, H);
    printf("Disvalue = %f\n", disvalue);
    
    nested_dict2 Quadtree;
    nested_dict Position;
    vector<float> ThresholdD;
    
    cout << "{";
    for( int i = 0; i < dmax; i++) {
        ThresholdD.push_back(0);
        cout << i << ": " << ThresholdD[i];
        i < dmax - 1 ? cout << ", " : cout << "}" << endl;
    }
    cout << "[" << 0+(D) << ", " << 0+(D) << ", " << 0+(D)+W << ", " << 0+(D)+H << "]\n";
    
//##############################################################
    float Global_thresh = threshold(&im, k, 0 + D, 0 + D + W, 0 + D, 0 + D + H);
    cout << Global_thresh << endl;
//####################################################################
    checkCell(&Quadtree, &Position, &ThresholdD, 0, k, Couple(0, 0), disvalue, smin, &im, W, H, crd, D, dmax);
    cout << "the time consumed to built the grid and multilevel thresholding is " << difftime(time(NULL), starttime) << endl;
    long Depth = Position.size();
    printf("Depths = %ld\n", Depth);
    
//############ weighting the average thresholding ####################
    for (int i = 0; i < ThresholdD.size(); i++) {
        if ( ThresholdD[i] != 0 )
            ThresholdD[i] = float(ThresholdD[i])/Quadtree[i].size();
        if ( i != 0 )
            ThresholdD[i] = float(ThresholdD[i] + Global_thresh)/2;
    }
    cout << endl;
    cout << "Depth  Average thresholding number of quadarnts " << endl;
    for (int i = 0; i < ThresholdD.size(); i++)
        printf(" %d         %f           %lu \n", i, ThresholdD[i], Quadtree[i].size());
    
    // refining the quadtree
    vector<Pos> Posit;
    for (int i = 0; i < ThresholdD.size(); i++) {
        for (int j = 0; j < Quadtree[i].size(); j++) {
            int keys;
            ((smin == 2) && (i == Depth-1)) ? keys = i - 1 : keys = i;
            if (Position[i][Quadtree[i][j]][0].Threshold >= pow(ThresholdD[keys], 2) / Global_thresh) {
                for (int m = 0; m < 9; m++) {
                    Posit.push_back(Position[i][Quadtree[i][j]][m]);
                }
            }
        }
    }

    // removing duplicates
    vector<DTriple> NoDubPosit;
    set<Couple> visited;

    for(int i = 1; i < Posit.size(); i++ ) {
        if (visited.find(Couple(Posit[i].x, Posit[i].y)) == visited.end()) {
            visited.insert(Couple(Posit[i].x, Posit[i].y));
            NoDubPosit.push_back(DTriple(Posit[i].x, Posit[i].y, 1));
        }
    }
    Posit.clear();
    visited.clear();
    ThresholdD.clear();
    
    // sorting the positions
    sort(NoDubPosit.begin(), NoDubPosit.end(), sortfunct);
    
    // ploting positions
    int magn = 3; // scale plot according to the original image size
    Mat img(magn * imHeight, magn * imWidth, CV_8U);
    Mat tim;
    im.convertTo(tim, CV_8U, 0.5, 125);
    resize(tim, img, img.size(), magn, magn, INTER_NEAREST);
    tim.release();
    cvtColor(img, img, COLOR_GRAY2RGB);
    Point pnt;
    for(int i = 1; i < NoDubPosit.size(); i++ ) {
        pnt.x = int(magn * NoDubPosit[i].k0);
        pnt.y = int(magn * NoDubPosit[i].k1);
        circle( img, pnt, 3, Scalar(255, 0, 0), -1);
    }
    imwrite(dir_outpu+sep+"plot_grid"+sep+"plot_positions.jpg", img);
    
    // generate grid
    ag = Generate_Edges_Convs(Depth, Couple(0,0), &im, disvalue, W, H, smin, &NoDubPosit, dir_conv);
    
    // ploting grid
    magn = 3; // scale plot according to the original image size
    im.convertTo(tim, CV_8U, 0.5, 125);
    resize(tim, img, img.size(), magn, magn, INTER_NEAREST);
    tim.release();
    cvtColor(img, img, COLOR_GRAY2RGB);
    Point pnt1, pnt2;
    int u, v;
    for(int i = 0; i < ag.Edges.size(); i++ ) {
        u = ag.Edges[i].k0;
        v = ag.Edges[i].k1;
        pnt1.x = int(magn * ag.pos[u].k0);
        pnt1.y = int(magn * ag.pos[u].k1);
        pnt2.x = int(magn * ag.pos[v].k0);
        pnt2.y = int(magn * ag.pos[v].k1);
        if (pnt1 != pnt2)
            line(img, pnt1, pnt2, Scalar(255, 0, 0), 2);
        else
            line(img, pnt1, pnt2, Scalar(0, 255, 0), 2);
    }
    imwrite(dir_outpu+sep+"plot_grid"+sep+"plot_grid.jpg", img);
    img.release();
    
    cout << "The adaptive grid is built!" << endl;
    return ag;
}
//---------------------------------------------------------------------------//
// edgekernel is Edges detection function using Gaussian Kernel
matrix edgekernel(int lx, int ly, double v, double x1, double y1, double x2, double y2, int pbcx, int pbcy) {
    vector<double> dx1, dx2, dy1, dy2;
    for (int i = 0; i < lx; i++) {
        dx1.push_back(pow(i - x1, 2) / (2.0*v));
        dx2.push_back(pow(i - x2, 2) / (2.0*v));
    }
    for (int i = 0; i < ly; i++) {
        dy1.push_back(pow(i - y1, 2) / (2.0*v));
        dy2.push_back(pow(i - y2, 2) / (2.0*v));
    }
    
    matrix ex1, ex2, ey1, ey2, ek;
    
    ek.resize(ly);
    
    for (int i = 0; i < ly; i++) {
        ek[i].resize(lx, 0);
        ex1.push_back(dx1);
        ex2.push_back(dx2);
    }
    for (int i = 0; i < lx; i++) {
        ey1.push_back(dy1);
        ey2.push_back(dy2);
    }
    
    double summ = 0;
    for (int i = 0; i < ly; i++) {
        for (int j = 0; j < lx; j++) {
            ek[i][j] = exp( -( sqrt(ex1[i][j] +  ey1[j][i]) + sqrt(ex2[i][j] + ey2[j][i]) ) );
            summ += ek[i][j];
        }
    }
    
    for (int i = 0; i < ly; i++)
        for (int j = 0; j < lx; j++)
            ek[i][j] /= summ;
    
    return ek;
}
//---------------------------------------------------------------------------//
igraph_t grid_grid_all(Mat im, string file_path, Adaptive_Grid * AG, int dz)
{
    vector<double> capas;
    vector<matrix> conv;
    
    int n, m;
    double x1, x2, y1, y2;

    for (int e = 0; e < AG->Edges.size(); e++) {
        n = AG->Edges[e].k0;
        m = AG->Edges[e].k1;
        x1 = AG->pos[n].k0;
        y1 = AG->pos[n].k1;
        x2 = AG->pos[m].k0;
        y2 = AG->pos[m].k1;
        conv.push_back( edgekernel(im.cols, im.rows, AG->Disvalue, x1, y1, x2, y2, 0, 0) );
//        save0(AG->dir_conv + "_L=" + to_string(e), row);
    }
    
    double csumm = 0;
    for (int e = 0; e < AG->Edges.size(); e++) { //loop over n# of edges
        double summ = 0;
        for (int i = 0; i < im.cols; i++)
            for (int j = 0; j < im.rows; j++)
                summ += im.at<double>(i,j) * conv[e][j][i];
        capas.push_back(summ);
        csumm += summ;
    }
    
    for(int e = 0; e < AG->Edges.size(); e++)
        capas[e] /= csumm;
    cout << "creating the graph\n";
    
    igraph_integer_t no = 0;
    igraph_integer_t gr_size = (int)AG->Edges.size();
    igraph_t graph;
//    igraph_vector_t edgs;
    igraph_vector_t wht;
    
    igraph_i_set_attribute_table(&igraph_cattribute_table);
    
    igraph_empty(&graph, (int)AG->pos.size(), 0);
//    igraph_vector_init(&edgs, 2*gr_size);
    igraph_vector_init(&wht, gr_size);

    for (int e = 0; e < gr_size; e++) {
        n = AG->Edges[e].k0;
        m = AG->Edges[e].k1;
        igraph_add_edge(&graph, n, m);

//        VECTOR(edgs)[2*e] = n;
//        VECTOR(edgs)[2*e + 1] = m;
        VECTOR(wht)[e] = capas[e];
        no++;
        printf("(%d, %d)\n", n, m);
    }
    
//    igraph_add_edges(&graph, &edgs, 0);
    SETEANV(&graph, "capa", &wht);

    printf("no. of graph edges = %d\n", no);

    igraph_integer_t diameter;
    igraph_diameter(&graph, &diameter, 0, 0, 0, IGRAPH_UNDIRECTED, 1);
    printf("Diameter of the graph: %d\n", (int) diameter);

    igraph_real_t cluster;
    igraph_transitivity_undirected(&graph, &cluster, IGRAPH_TRANSITIVITY_NAN);
    printf("Clustering coefficient: %f\n", (float) cluster);

//    igraph_vector_t degree;
//    igraph_degree(&graph, &degree, igraph_vss_all(), IGRAPH_ALL, IGRAPH_NO_LOOPS);
//    printf("mean[degree] = %f\n" , (float)igraph_vector_prod(&degree) / (float)igraph_vector_size(&degree) );
    
    return graph;
}
//---------------------------------------------------------------------------//
Data_Label graph_graph_all(Graph G,std::vector<Triple> pos)
{
    Data_Label data_label;
    return data_label;
}
//---------------------------------------------------------------------------//
void plot0(string url, int imHeight, int imWidth, std::vector<Triple> pos)
{
    int ly = imHeight;
    int lx = imWidth;
    Graph gn=load1(url);
    int N=number_of_nodes(gn);
    //Return a copy of the graph gn with the nodes relabeled using consecutive integers.
    Graph gc=convert_node_labels_to_integers(subgraph(gn,N-N/1,N));
    //posi=load(dir_output+sep+"data_posi"+sep+"data_posi.npy").flatten()[0]
//    Pos2D3D pos2Dpos3D=image_graph_help_grid_pos2D(1,pos);
    // en=numpy.array([d['capa'] for u,v,d in gn.edges_iter(data=1)])
    // en=en/en.max()
    // ec=numpy.array([d['capa'] for u,v,d in gc.edges_iter(data=1)])
    // ec=ec/en.max()
    // matplotlib.pyplot.clf()
}
//---------------------------------------------------------------------------//
void Pos2D3D_image_graph_help_grid_pos2D(int n,std::vector<Triple> pos)
{
}
//---------------------------------------------------------------------------//
void write0(string url,Labels labels)
{
}
//---------------------------------------------------------------------------//
void save0(string url,std::vector<Triple> Edges)
{
}
//---------------------------------------------------------------------------//
void save0(string url, std::vector<double> data) {
    ofstream file;
    file.open (url, ios::ate);
    for (auto it = data.begin(); it != data.end(); it++)
        file << to_string(*it) << "\t";
    file <<endl;
    file.close();
}
//---------------------------------------------------------------------------//
void save0(string url,string label)
{
}
//---------------------------------------------------------------------------//
Labels load0(string url)
{
    Labels labels;
    return labels;
}
//---------------------------------------------------------------------------//
Graph load1(string url)
{
    Graph g;
    return g;
}
//---------------------------------------------------------------------------//
int number_of_nodes(Graph g)
{
    int n;
    return n;
}
//---------------------------------------------------------------------------//
Graph subgraph(Graph g,int n1, int n2)
{
    Graph g1;
    return g1;
}
//---------------------------------------------------------------------------//
Graph convert_node_labels_to_integers(Graph g)
{
    Graph g1;
    return g1;
}
//---------------------------------------------------------------------------//
Pos2D3D convert_node_labels_to_integers(int nz, std::vector<Triple> pos)
{
    Pos2D3D p;
    return p;
}
//---------------------------------------------------------------------------//
int main( int argc, char** argv )
{
    image_graph_run();
    return 0;
}