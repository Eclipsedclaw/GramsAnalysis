#define ana_cxx
#include "ana.h"
#include <TH2.h>
#include <TF1.h>
#include <TStyle.h>
#include <TCanvas.h>
Double_t ff1(Double_t *x, Double_t *par){

  double t=x[0];
  double q=par[0];
  double t0=par[1];
  double t1=par[2];
  double tau_amp = par[3];
  double tau_imp = par[4];

    if (t <= t0) return 0;
    else if (t > t0 && t <= t1 && tau_amp != tau_imp)
        return (1 / (t1 - t0)) * q * ((tau_amp * tau_imp) / (tau_amp - tau_imp)) * (exp(-(t - t0) / tau_amp) - exp(-(t - t0) / tau_imp));
    else if (t > t1 && tau_amp != tau_imp)
        return (1 / (t1 - t0)) * q * ((tau_amp * tau_imp) / (tau_amp - tau_imp)) * (exp(-(t1 - t0) / tau_amp) - exp(-(t1 - t0) / tau_imp)) * exp(-(t - t1) / tau_amp);
    else
        return 0;

}

void ana::Loop()
{
//   In a ROOT session, you can do:
//      root> .L ana.C
//      root> ana t
//      root> t.GetEntry(12); // Fill t data members with entry number 12
//      root> t.Show();       // Show values of entry 12
//      root> t.Show(16);     // Read and show values of entry 16
//      root> t.Loop();       // Loop on all entries
//

//     This is the loop skeleton where:
//    jentry is the global entry number in the chain
//    ientry is the entry number in the current Tree
//  Note that the argument to GetEntry must be:
//    jentry for TChain::GetEntry
//    ientry for TTree::GetEntry and TBranch::GetEntry
//
//       To read only selected branches, Insert statements like:
// METHOD1:
//    fChain->SetBranchStatus("*",0);  // disable all branches
//    fChain->SetBranchStatus("branchname",1);  // activate branchname
// METHOD2: replace line
//    fChain->GetEntry(jentry);       //read all branches
//by  b_branchname->GetEntry(ientry); //read only this branch
   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   int lch[60];
   for (int i=0;i<60;i++) {lch[i] = {-999};}
   int xch[60];
   for (int i=0;i<60;i++) {xch[i] = {-999};}
   int ych[60];
   for (int i=0;i<60;i++) {ych[i] = {-999};}

   lch[2]=2;
   lch[3]=3;
   lch[19]=1;
   lch[35]=0;


   xch[5]=2;
   xch[6]=5;
   xch[7]=8;
   xch[8]=11;
   xch[9]=14;
   xch[21]=1;
   xch[22]=4;
   xch[23]=7;
   xch[24]=10;
   xch[25]=13;
   xch[26]=15;
   xch[38]=0;
   //xch[39]=3;
   xch[56]=3;
   xch[40]=6;
   xch[41]=9;
   xch[42]=12;

   ych[10]=2;
   ych[11]=5;
   ych[12]=8;
   ych[13]=11;
   ych[14]=14;
   ych[27]=1;
   ych[28]=4;
   ych[29]=7;
   ych[30]=10;
   ych[31]=13;
   ych[43]=0;
   ych[44]=3;
   ych[45]=6;
   ych[46]=9;
   ych[47]=12;
   ych[48]=15;

   
   TH1D *hl[4];
   for (int i=0;i<4;i++) {hl[i] = new TH1D(Form("hl%i",i),Form("hl%i",i),37500,-15,285);}
   TH1D *hx[16];
   for (int i=0;i<16;i++) {hx[i] = new TH1D(Form("hx%i",i),Form("hx%i",i),37500,-15,285);}
   TH1D *hy[16];
   for (int i=0;i<16;i++) {hy[i] = new TH1D(Form("hy%i",i),Form("hy%i",i),37500,-15,285);}

   double pedl[4]={0.};
   double pedx[16]={0.};
   double pedy[16]={0.};
   
   TH1D *hlsum = new TH1D("hlsum","hlsum",37500,-16,284);
   TH1D *hxsum = new TH1D("hxsum","hxsum",37500,-16,284);
   TH1D *hysum = new TH1D("hysum","hysum",37500,-16,284);
   TH1D *hxysum = new TH1D("hxysum","hxysum",37500,-16,284);

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      // if (Cut(ientry) < 0) continue;
      if (lch[channel] > -1){
	int ich=lch[channel];
	if (event_id<10) printf("Light: %i %i %i\n",event_id,channel,ich);
	for (int i=0; i<37500; i++) {
	  hl[ich]->AddBinContent(i+1,-waveform_samples[i]/1000.);
	  hlsum->AddBinContent(i+1,-waveform_samples[i]/1000.);
	  if (i<1250) pedl[ich]=pedl[ich]-waveform_samples[i]/1000./1250.;
	}
      }
      else if (xch[channel] > -1){
	int ich=xch[channel];
	if (event_id<10) printf("X   : %i %i %i\n",event_id,channel,ich);
	for (int i=0; i<37500; i++) {
	  hx[ich]->AddBinContent(i+1,waveform_samples[i]/1000.);
	  hxsum->AddBinContent(i+1,waveform_samples[i]/1000.);
	  hxysum->AddBinContent(i+1,waveform_samples[i]/1000.);
	  if (i<1250) pedx[ich]=pedx[ich]+waveform_samples[i]/1000./1250.;
	}
      }
      else if (ych[channel] > -1){
	int ich=ych[channel];
	if (event_id<10) printf("Y   : %i %i %i\n",event_id,channel,ich);
	for (int i=0; i<37500; i++) {
	  hy[ich]->AddBinContent(i+1,waveform_samples[i]/1000.);
	  hysum->AddBinContent(i+1,waveform_samples[i]/1000.);
	  hxysum->AddBinContent(i+1,waveform_samples[i]/1000.);
	  if (i<1250) pedy[ich]=pedy[ich]+waveform_samples[i]/1000./1250.;	}
      }
      //break;
   }
   
   for (int j=0;j<37500;j++){
      for (int i=0;i<4;i++)  {
	hl[i]->AddBinContent(j+1,-pedl[i]);
	hlsum->AddBinContent(j+1,-pedl[i]);
      }
      for (int i=0;i<16;i++) {
	hx[i]->AddBinContent(j+1,-pedx[i]);
	hy[i]->AddBinContent(j+1,-pedy[i]);
	hxsum->AddBinContent(j+1,-pedx[i]);
	hysum->AddBinContent(j+1,-pedy[i]);
	hxysum->AddBinContent(j+1,-pedx[i]);
	hxysum->AddBinContent(j+1,-pedy[i]);
      }
   }

   hxsum->Rebin(125);hxsum->Scale(1./125);hxsum->SetMaximum(10);
   hysum->Rebin(125);hysum->Scale(1./125);hysum->SetMaximum(10);
   hxysum->Rebin(125);hxysum->Scale(1./125);hxysum->SetMaximum(10);
   hlsum->GetXaxis()->SetRangeUser(-1,10);
   hxsum->GetXaxis()->SetRangeUser(-15,150);
   hysum->GetXaxis()->SetRangeUser(-15,150);
   hxysum->GetXaxis()->SetRangeUser(-15,150);

   hlsum->SetXTitle("Time (#mus)");
   hlsum->SetYTitle("SiPM Summed WF (mV)");
   hxsum->SetXTitle("Time (#mus)");
   hxsum->SetYTitle("X ch Summed WF (mV)");
   hysum->SetXTitle("Time (#mus)");
   hysum->SetYTitle("Y ch Summed WF (mV)");
   hxysum->SetXTitle("Time (#mus)");
   hxysum->SetYTitle("X+Y ch Summed WF (mV)");
   
   TCanvas *c1 = new TCanvas("c1","c1",800,800);
   c1->Divide(2,2);
   c1->cd(1);
   hlsum->Draw();
   c1->cd(2);
   hxysum->Draw();   
   c1->cd(3);
   hxsum->Draw();
   c1->cd(4);
   hysum->Draw();


   TF1 *f1 = new TF1("f1",ff1,5,150,5);
   TF1 *f2 = new TF1("f2",ff1,5,150,5);
   TF1 *f3 = new TF1("f3",ff1,5,150,5);

   f1->SetLineColor(1);
   f2->SetLineColor(2);
   f3->SetLineColor(4);
   
   TCanvas *c2 = new TCanvas("c2","c2",800,800);
   hxysum->Draw();
   f1->SetParameters(80,5,100,500,10);
   f1->Draw("same");
   f2->SetParameters(11,5,20,15,500);
   f2->Draw("same");
   f3->SetParameters(150,5,100,15,10);
   f3->Draw("same");

   c1->Print("c1.png");
   c2->Print("c2.png");

}
