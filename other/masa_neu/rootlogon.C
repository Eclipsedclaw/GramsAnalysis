{

//ANKOK TStyle
TStyle* ANKOKStyle = new  TStyle("ANKOKStyle", "ANKOK Style");

//set the background color to white
ANKOKStyle->SetFillColor(10);
ANKOKStyle->SetFrameFillColor(10);
ANKOKStyle->SetCanvasColor(10);
ANKOKStyle->SetPadColor(10);
ANKOKStyle->SetTitleFillColor(0);
ANKOKStyle->SetStatColor(10);

//dont put a colored frame around the plots
ANKOKStyle->SetFrameBorderMode(0);
ANKOKStyle->SetCanvasBorderMode(0);
ANKOKStyle->SetPadBorderMode(0);
ANKOKStyle->SetLegendBorderSize(0);

//use the primary color palette
ANKOKStyle->SetPalette(1,0);

//set the default line color for a histogram to be black
ANKOKStyle->SetHistLineColor(kBlack);

//set the default line color for a fit function to be red
ANKOKStyle->SetFuncColor(kRed);

//make the axis labels black
ANKOKStyle->SetLabelColor(kBlack,"xyz");

//set the default title color to be black
ANKOKStyle->SetTitleColor(kBlack);
 
//set the margins
// ANKOKStyle->SetPadBottomMargin(0.15);
// ANKOKStyle->SetPadTopMargin(0.10);
// ANKOKStyle->SetPadRightMargin(0.17);
// ANKOKStyle->SetPadLeftMargin(0.17);

//set the margins
ANKOKStyle->SetPadBottomMargin(0.15);
ANKOKStyle->SetPadTopMargin(0.08);
ANKOKStyle->SetPadRightMargin(0.1);
ANKOKStyle->SetPadLeftMargin(0.15);

//set axis label and title text sizes
ANKOKStyle->SetLabelFont(42,"xyz");
ANKOKStyle->SetLabelSize(0.06,"xyz");
ANKOKStyle->SetLabelOffset(0.015,"xyz");
ANKOKStyle->SetTitleFont(42,"xyz");
ANKOKStyle->SetTitleSize(0.06,"xz");
ANKOKStyle->SetTitleSize(0.06,"y");
ANKOKStyle->SetTitleOffset(1.2,"x");
ANKOKStyle->SetTitleOffset(1.2,"y");
ANKOKStyle->SetTitleOffset(1.5,"z");
ANKOKStyle->SetStatFont(42);
ANKOKStyle->SetStatFontSize(0.05);
ANKOKStyle->SetTitleBorderSize(0);
ANKOKStyle->SetStatBorderSize(0);
ANKOKStyle->SetTextFont(42);

//set line widths
ANKOKStyle->SetFrameLineWidth(2);
ANKOKStyle->SetFuncWidth(2);
ANKOKStyle->SetHistLineWidth(2);

//set the number of divisions to show
ANKOKStyle->SetNdivisions(506, "xy");

//turn off xy grids
ANKOKStyle->SetPadGridX(1);
ANKOKStyle->SetPadGridY(1);

//set the tick mark style
ANKOKStyle->SetPadTickX(1);
ANKOKStyle->SetPadTickY(1);

//turn off stats
ANKOKStyle->SetOptStat(0);
ANKOKStyle->SetOptFit(1);
ANKOKStyle->SetOptTitle(1);

//marker settings
//ANKOKStyle->SetMarkerStyle(20);
//ANKOKStyle->SetMarkerSize(0.5);
//ANKOKStyle->SetLineWidth(2); 

//done
ANKOKStyle->cd();
gROOT->ForceStyle();
gStyle->ls();

}

