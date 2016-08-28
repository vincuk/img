// [includes]
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#include <ctime>
#include <vector>
#include <queue>
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
struct Adaptive_Grid{
    std::vector<Triple> Edges;
    std::vector<Triple> pos;
    string dir_conv;
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
typedef map< int, vector<Couple> > nested_dict2;
typedef map< Couple, vector<Pos> > mapelement;
typedef map< int,  mapelement> nested_dict;
//---------------------------------------------------------------------------//
void image_graph(int imWidth,int imHeight,string crd, Mat image, int smin, int thresholding_m);
void image_graph_run();
void image_graph_calc(string crd,string dir_input,string file_input);
Adaptive_Grid image_graph_AMR_2D_Adaptive_grid(int imWidth,int imHeight, string crd, Mat im, string dir_conv,string dir_Edges,string dir_QuadTree,string dir_output);
Graph grid_grid_all(Mat im, string file_path,Adaptive_Grid Edges_pos_dir_conv,int dz);
void save0(string url,std::vector<Triple> Edges);
void save0(string url,std::vector<int> data);
void save0(string url,string label);
Data_Label graph_graph_all(Graph G,std::vector<Triple> pos);
Labels load0(string url);
void write0(string url,Labels labels);
void plot0(string url, int imHeight, int imWidth, std::vector<Triple> pos);
Labels load0(string url);
Graph load1(string url);
int number_of_nodes(Graph g);
Graph subgraph(Graph g,int n1, int n2);
Graph convert_node_labels_to_integers(Graph g);
Pos2D3D image_graph_help_grid_pos2D(int n,std::vector<Triple> pos);
vector<float> ThresholdD;

//---------------------------------------------------------------------------//
void image_graph_run()
{
    string dir_input("/Users/vincUk/Desktop/image_graph/images/");
//    string file_name("Control1.tif");
    string file_name("Cytoskeletal-1.tif");
    string gridtype("rectangular");
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
    Adaptive_Grid Edges_pos_dir_conv = image_graph_AMR_2D_Adaptive_grid(imWidth, imHeight, crd, im, dir_conv, dir_Edges, dir_QuadTree, dir_output);
    cout<<"graph"<<endl;
    time_t temp1 = time(0);
    Graph graph = grid_grid_all(im,file_path,Edges_pos_dir_conv,1);
    time_t temp2 = time(0);
    save0(dir_output+sep+"data_grph"+sep+"data_grph.npy",graph.Edges);
    
    cout << "the time consumed to build the graph is " << (temp2-temp1) << endl;
    cout<< "obs network"<<endl;
    
    temp1 = time(0);
    Data_Label data_label = graph_graph_all(graph,Edges_pos_dir_conv.pos);
    temp2 = time(0);
    cout<<"the time consumed to measure the graph's properties quantitatively is "<<(temp2-temp1)<<endl;
    save0(dir_output+sep+"data_datn"+sep+"data_datn.npy",data_label.data);
    save0(dir_output+sep+"data_prop"+sep+"data_prop.npy",data_label.label);
    
    Labels labels=load0(dir_output+sep+"data_prop"+sep+"data_prop.npy");
    
    write0(dir_output+sep+"data_readable"+sep+"data_readable.txt",labels);
    // with open(dir_output"+sep+"data_readable"+sep+"data_readable.txt","a") as out:
    //     out.write('\t'.join([str(a) for a in numpy.hstack(labels)]))
    //     out.write('\n')
    //     datn=numpy.load(os.path.join(dir_output,'data_datn','data_datn.npy'))
    //     out.write('\t'.join([str(a) for a in numpy.hstack(datn)]))
    //     out.write('\n')
    
    plot0(dir_output+sep+"data_grph"+sep+"data_grph.npy",imHeight,imWidth,Edges_pos_dir_conv.pos);
}
//---------------------------------------------------------------------------//
float threshold_otsu(Mat * im) {
    return 0;
}
float treshold(Mat * img, int k, int x1, int x2, int y1, int y2) {
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
    float B1_thresh = treshold(im, k, posi->operator[](0).x, posi->operator[](4).x, posi->operator[](0).y, posi->operator[](4).y);
    float B2_thresh = treshold(im, k, posi->operator[](1).x, posi->operator[](5).x, posi->operator[](1).y, posi->operator[](5).y);
    float B3_thresh = treshold(im, k, posi->operator[](3).x, posi->operator[](7).x, posi->operator[](3).y, posi->operator[](7).y);
    float B4_thresh = treshold(im, k, posi->operator[](4).x, posi->operator[](8).x, posi->operator[](4).y, posi->operator[](8).y);
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
int Divide_Decision(nested_dict * PS, int Depth, int k, Couple cellCoords, float disvalue, float smin, int dmax, Mat * im, int imWidth, int imHeight, string crd, int D) {
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
        float B_thresh1 = treshold(im, k, x1, x2, y1, y2);
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
            float AddedValue = ThresholdD[Depth] + avgDthreshold;
            ThresholdD[Depth] = AddedValue;
            printf("the sub-quads threshoding = %f \n", avgDthreshold);
            return 0;
        }
        else {
            if (int(mind) >= 2) {
                float avgDthreshold = QudtreeThreshold(im, k, &posi, Depth);
                float AddedValue= ThresholdD[Depth]+ avgDthreshold;
                ThresholdD[Depth] = AddedValue;
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
void checkCell(nested_dict2 * QT, nested_dict * PS, int Depth, int k, Couple cellCoords, float disvalue, float smin, Mat * im, int imWidth, int imHeight, string crd, int D, int dmax) {
    if (Depth > dmax+1) {
        cout << "the grid reaches the maximum depth " << Depth << endl;
        return;
    }
    //# Dividing the current depth for 4 parts :
    int minThreshold = Divide_Decision(PS, Depth, k, cellCoords, disvalue, smin, dmax, im, imWidth, imHeight, crd, D);
    if (minThreshold  != 1) {
        if (QT->find(Depth) == QT->end()) {
            vector<Couple> v = {cellCoords};
            QT->insert(pair<int, vector<Couple> >(Depth,v));
        }
        else
            QT->at(Depth).push_back(cellCoords);
        if (minThreshold  != 2) {
            checkCell(QT, PS, Depth + 1, k, dblCouple(cellCoords), disvalue, smin, im, imWidth, imHeight, crd, D, dmax);
            checkCell(QT, PS, Depth + 1, k, dblCouple(cellCoords) + Couple(0,1), disvalue, smin, im, imWidth, imHeight, crd, D, dmax);
            checkCell(QT, PS, Depth + 1, k, dblCouple(cellCoords) + Couple(1,0), disvalue, smin, im, imWidth, imHeight, crd, D, dmax);
            checkCell(QT, PS, Depth + 1, k, dblCouple(cellCoords) + Couple(1,1), disvalue, smin, im, imWidth, imHeight, crd, D, dmax);
        }
    }
    
};
//---------------------------------------------------------------------------//
bool sortfunct (Pos i, Pos j) {
    if (i.x < j.x && i.y < j.y) return true;
    if (i.x == j.x && i.y < j.y) return true;
    return false;
}
//---------------------------------------------------------------------------//
Triple most_common(vector<Triple> list) {
    map<Triple, int> mp;
    int maxv = 0;
    Triple com = list[0];
    for (vector<Triple>::iterator it = list.begin(); it != list.end(); it++) {
        if (mp.find(*it) == mp.end()) {
            mp[*it] = 1;
        }
        else {
            mp[*it]++;
            if (mp[*it] > maxv) {
                maxv = mp[*it];
                com = *it;
            }
        }
    }
    return com;
}
//---------------------------------------------------------------------------//
Adaptive_Grid Generate_Edges_Convs(long Depth, Couple cellCoords, Mat * im, float DisValue, int imWidth,int imHeight, int MinSize, vector<Triple> * Posit, string dir_conv) {
    Adaptive_Grid ag;
    float dx = float(imWidth) / pow(2, Depth);
    float dy = float(imHeight) / pow(2, Depth);
    
    //#################### sorting the positions ######################
//    sort(Posit.begin(), Posit.end(), sortfunct);


//######################################################################
//# Generating the Edges
    vector<Triple> Edges;
    for (int key1 = 0; key1 < Posit->size(); key1++) {
        for (int key2 = 0; key2 < key1; key2++) {
            int Dx = fabs(Posit->operator[](key1).k0 - Posit->operator[](key2).k0);
            int Dy = fabs(Posit->operator[](key1).k0 - Posit->operator[](key2).k1);
            if (pow(float(Dx)/dx, 2) + pow(float(Dy)/dy, 2) < 1.1)
                Edges.push_back(Triple(key1, key2, 1));
        }
    }
    
//########################### Finding Edges' nodes in the main grid #######################
    vector<int> connectedNodes;
    Triple z1 = most_common(Edges);
    queue<int> Q;
    Q.push(z1.k0);
    set<int> visited;
    
    return ag;
}
//---------------------------------------------------------------------------//
Adaptive_Grid image_graph_AMR_2D_Adaptive_grid(int imWidth,int imHeight, string crd, Mat im, string dir_conv, string dir_Edges, string dir_QuadTree, string dir_outpu)
{
    Adaptive_Grid ag;
    
    time_t temp = time(0);
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
    
    cout << "{";
    for( int i = 0; i < dmax; i++) {
        ThresholdD.push_back(0);
        cout << i << ": " << ThresholdD[i];
        i < dmax - 1 ? cout << ", " : cout << "}" << endl;
    }
    cout << "[" << 0+(D) << ", " << 0+(D) << ", " << 0+(D)+W << ", " << 0+(D)+H << "]\n";
    
//##############################################################
    float Global_thresh = treshold(&im, k, 0 + D, 0 + D + W, 0 + D, 0 + D + H);
    cout << Global_thresh << endl;
//####################################################################
    nested_dict2 Quadtree;
    nested_dict Position;

    Couple coord(0, 0);
    checkCell(&Quadtree, &Position, 0, k, coord, disvalue, smin, &im, W, H, crd, D, dmax);
    cout << "the time consumed to built the grid and multilevel thresholding is " << (time(0) - temp) << endl;
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
    
//################# refining the quadtree ##############################
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

    vector<Triple> NoDubPosit;
    set<Couple> seen;

    for(int i = 1; i < Posit.size(); i++ ) {
        if (seen.find(Couple(Posit[i].x, Posit[i].y)) == seen.end()) {
            seen.insert(Couple(Posit[i].x, Posit[i].y));
            NoDubPosit.push_back(Triple(Posit[i].x, Posit[i].y, Posit[i].Depth));
        }
    }
    Posit.clear();
    seen.clear();
    
    int magn = 3;
    Mat img(magn * imHeight, magn * imWidth, CV_8U);
    Mat tim;
    im.convertTo(tim, CV_8U, 0.5, 125);
    resize(tim, img, img.size(), magn, magn, INTER_NEAREST);
    tim.release();
    cvtColor(img, img, COLOR_GRAY2RGB);
    Point pnt;
    for(int i = 1; i < NoDubPosit.size(); i++ ) {
        pnt.x = magn * NoDubPosit[i].k0;
        pnt.y = magn * NoDubPosit[i].k1;
        circle( img, pnt, 3, Scalar(255, 0, 0), -1);
    }
    //////////
    imwrite(dir_outpu+sep+"plot_grid"+sep+"plot_positions.jpg", img);
    img.release();
    
    ag = Generate_Edges_Convs(Depth, Couple(0,0), &im, disvalue, W, H, smin, &NoDubPosit, dir_conv);
    return ag;
}
//---------------------------------------------------------------------------//
Graph grid_grid_all(Mat im, string file_path, Adaptive_Grid Edges_pos_dir_conv,int dz)
{
    Graph graph;
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
void save0(string url,std::vector<int> data)
{
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