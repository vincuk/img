#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <igraph.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#include <ctime>
//---------------------------------------------------------------------------//
//[namespace]
using namespace cv;
using namespace std;
//---------------------------------------------------------------------------//
#define sep "/"
//---------------------------------------------------------------------------//
struct Couple{
	int k0;
	int k1;
};
//---------------------------------------------------------------------------//
struct Triple{
	int k0;
	int k1;
	int k2;
};
//---------------------------------------------------------------------------//
struct Adaptive_Grid{
	std::vector<Triple> Edges;
	std::vector<Triple> pos;
	string dir_conv;
};
//---------------------------------------------------------------------------//
struct Graph{
	std::vector<Triple> Edges;
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
//---------------------------------------------------------------------------//
void image_graph(int imWidth,int imHeight,string crd, Mat image, int smin, int thresholding_m);
void image_graph_run();
void image_graph_calc(string crd,string dir_input,string file_input);
Adaptive_Grid image_graph_AMR_2D_Adaptive_grid(int imWidth,int imHeight,string crd,Mat im,string dir_conv,string dir_Edges,string dir_QuadTree,string dir_output);
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
//---------------------------------------------------------------------------//
void image_graph(int imWidth,int imHeight,string crd, Mat image, int smin, int thresholding_m)
{
    // timing
    printf("image width = %d, image height = %d\n", imWidth, imHeight);
 
    if (smin<2)
    {
        printf(" The minimum block size can not be less than 2 pixels \n");
        return;
    }           
    int D = log(min(imWidth, imHeight)/smin)+float(3/2);
    printf("The margins of the grid will be %d\n", D);
    int w = imWidth - (D*2);
    int h = imHeight - (D*2);
    printf("The Width of the grid = %d ,  and the Height of the grid = %d\n",w,h);
    int dmax = log(min(w, h)/smin) + float(3/2);
    printf("The maximum depth the grid could be divided into = %d\n", dmax);
    if (smin<=16) // set Guassian kernel
    {
        float v = 0.5*smin;
    }
    else
    {    
        float v= 8;
    }
    
    return;    
}
//---------------------------------------------------------------------------//
void image_graph_run()
{
    string dir_input("/home/next/Desktop/C/UpWork/PeterBrams/image_graph/images/");
    string file_name("Control1.tif");
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
	Adaptive_Grid Edges_pos_dir_conv=image_graph_AMR_2D_Adaptive_grid(imWidth,imHeight,crd,im,dir_conv,dir_Edges,dir_QuadTree,dir_output);
    cout<<"graph"<<endl;
    time_t temp1 = time(0); 
    Graph graph = grid_grid_all(im,file_path,Edges_pos_dir_conv,1); 
    save0(dir_output+sep+"data_grph"+sep+"data_grph.npy",graph.Edges);
    cout<<temp1<<"the time consumed to build the graph is "<<(time(0)-temp1)<<endl;
    cout<< "obs network"<<endl;

    time_t temp2 = time(0); 
    Data_Label data_label = graph_graph_all(graph,Edges_pos_dir_conv.pos);
    cout<<temp1<<"the time consumed to measure the graph's properties quantitatively is "<<(time(0)-temp2)<<endl;
    save0(dir_output+sep+"data_datn"+sep+"data_datn.npy",data_label.data);
	save0(dir_output+sep+"data_prop"+sep+"data_prop.npy",data_label.label);

    Labels labels=load0(dir_output+sep+"data_prop"+sep+"data_prop.npy");

    write0(dir_output+sep+"data_readable"+sep+"data_readable.txt",labels);

    plot0(dir_output+sep+"data_grph"+sep+"data_grph.npy",imHeight,imWidth,Edges_pos_dir_conv.pos);
}
//---------------------------------------------------------------------------//
Adaptive_Grid image_graph_AMR_2D_Adaptive_grid(int imWidth,int imHeight,string crd,Mat im,string dir_conv,string dir_Edges,string dir_QuadTree,string dir_output)
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
    int dmax = int(ceil(log(float(min(W, H))/smin)+float(3/2)));
    float disvalue;
    smin <= 16 ? disvalue= 0.5*smin : disvalue = 8;
    printf("smin = %d, dmax = %d, Distance = %d\n", int(smin), dmax, D);
    printf("image width = %d, image height = %d\n", W, H);
    printf("Disvalue = %f\n", disvalue);
    
    vector<float> ThresholdD;
    cout << "{";
    for( int i = 0; i < dmax; i++) {
        ThresholdD.push_back(0);
        cout << i << ": " << ThresholdD[i];
        i < dmax - 1 ? cout << ", " : cout << "}" << endl;
    }
    cout << "[" << 0+(D) << ", " << 0+(D) << ", " << 0+(D)+W << ", " << 0+(D)+H << "]\n";

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
    Pos2D3D pos2Dpos3D=image_graph_help_grid_pos2D(1,pos);
}
//---------------------------------------------------------------------------//
Pos2D3D image_graph_help_grid_pos2D(int n,std::vector<Triple> pos)
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
//---------------------------------------------------------------------------//