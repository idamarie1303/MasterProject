

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/FastJet3.h"
#include "fastjet/tools/JetMedianBackgroundEstimator.hh"
#include "fjcontrib/ConstituentSubtractor/ConstituentSubtractor.hh"
//Importing ROOT
#include "TH1F.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TH2F.h"
#include "TLatex.h"
#include <TStyle.h>
//Get bessel functions
#include <boost/math/special_functions/bessel.hpp>
//Other stuff
#include <time.h>
#include <iostream>
#include <vector>
#include <random>
#include <stdlib.h>
#include <numeric>
#include <algorithm>

using namespace Pythia8;


//Observables
auto get_m(vector <fastjet::PseudoJet> particlelist) -> std::tuple<double, double, double, double>{
  double E1 = particlelist[0].E();
  double E2 = particlelist[1].E();
  double px1 = particlelist[0].px();
  double px2 = particlelist[1].px();
  double py1 = particlelist[0].py();
  double py2 = particlelist[1].py();
  double pz1 = particlelist[0].pz();
  double pz2 = particlelist[1].pz();

  double p1_2 = E1*E1-px1*px1-py1*py1-pz1*pz1;
  double p2_2 = E2*E2-px2*px2-py2*py2-pz2*pz2;
  double p12_2 = E1*E2-px1*px2-py1*py2-pz1*pz2;

  double mh = p1_2+p2_2+2*p12_2;
  double m0 = 2*p12_2;

  double z = min(particlelist[0].pt(),particlelist[1].pt())/(particlelist[0].pt()+particlelist[1].pt());

  double kt_2 = z*(1-z)*mh;

  return {mh, m0, z, kt_2};
}

float mdelta(fastjet::PseudoJet jet){
    float md = sqrt(pow(jet.m(),2)+pow(jet.pt(),2))-jet.pt();
    return md;
}


//Background
double_t myFunc(float p_T, float beta_T, float T_F, float A_i) { 
  float rho = atanh(beta_T);
  float inK = p_T*cosh(rho) / T_F;
  float inI = p_T*sinh(rho) / T_F;

  float I0 = boost::math::cyl_bessel_i(0, inI);
  float K1 = boost::math::cyl_bessel_k(1, inK);

  return A_i*p_T*K1*I0;
}

vector <fastjet::PseudoJet> getbackground(float N, float beta_T, float T_F, float max_f) {

  float A_i = 1;
  float maxeta = 2.5;

  std::mt19937 generator(time(nullptr));
  std::uniform_real_distribution <double>  distr(0, 2*M_PI);
  std::uniform_real_distribution <double>  distr2(-maxeta, maxeta);
  std::uniform_real_distribution <double>  distr3(0, 1000.);
  std::uniform_real_distribution <double>  distr4(0, 1.);

  vector <fastjet::PseudoJet> listjets;
  for (int i = 0; i < N; i++){
    double phi = distr(generator);
    double eta = distr2(generator);
    float pt;
    bool done=false;
    while (done==false){
      double rand3 = distr3(generator);
      double rand4 = distr4(generator);
      if (rand4 > myFunc(rand3, beta_T, T_F, A_i)/max_f) {
        continue;
      } else{
        pt = rand3;
        done=true;
      }
    }
    
    //Create 4-vector (frmula (2.32) in 1901.10342)
    float E  = pt*cosh(eta)*pow(10,-3);
    float px = pt*cos(phi)*pow(10,-3);
    float py = pt*sin(phi)*pow(10,-3);
    float pz = pt*sinh(eta)*pow(10,-3);

    fastjet::PseudoJet jet = fastjet::PseudoJet(px, py, pz, E);

    listjets.push_back(jet);
  }

  // Done.
  return listjets;
}

//Q-jet
vector <float> get_dij(vector <fastjet::PseudoJet> particles, double R, double power) {
  vector < float> deltaR ;
  vector < float> dij ;
  //cout << particles.size()<< endl;
  for (int i = 0; i < particles.size(); ++i) {
    for (int j = 0; j < particles.size(); ++j) {
      if (i >= j) continue;
      //cout << i << "--" << j <<endl;
      double dR = particles[i].delta_R(particles[j]); 
      
      //cout << min(pow(particles[i].pt(), -2),pow(particles[i].pt(), -2))*pow(dR,2)/pow(R,2) << endl;
      dij.push_back(min(pow(particles[i].pt(), 2*power),pow(particles[j].pt(), 2*power))*pow(dR,2)/pow(R,2));
    }
  }
  return dij;
}

class MyClass {             // The class where I store the jets
  public:             
    int id;                 // Id > 0
    int d1;                 // Id of Child1 
    int d2;                 // Id of Child2
    fastjet::PseudoJet jet; // Jet belonging to Id
    int nrparticles;

    MyClass(int ids, int d1s, int d2s, fastjet::PseudoJet jets, int nr){
      id  = ids;
      d1  = d1s;
      d2  = d2s;
      jet = jets;
      nrparticles = nr;
    }
};

auto GetIndexForClustering(vector <fastjet::PseudoJet> new_particles, float R, int power, float alpha)->std::tuple<int, int>{

    vector <float> dij = get_dij(new_particles, R, power);

    vector<int> index_1;
    vector<int> index_2;

    for (int i = 0; i < new_particles.size(); ++i) {
        for (int j = 0; j < new_particles.size(); ++j) {
            if (i >= j) continue;
            index_1.push_back(i);
            index_2.push_back(j);
        }
    }
    
    float d_min = *min_element(dij.begin(), dij.end()) ;

    vector <float> wij;
    float w_sum = 0;
    for (int d = 0; d< dij.size(); ++d){
        wij.push_back(exp(-alpha*(dij[d]-d_min)/d_min));
        w_sum+=wij[d];
    }

    vector <float> Pij;
    float sumP = 0;
    for(int i=0;i<wij.size();++i){
        Pij.push_back( wij[i] / w_sum);
        sumP+=Pij[i];
    }

    float rnum = (float) rand()/RAND_MAX;

    int index;
    float p_sum = 0;
    for (int i=0; i<Pij.size(); i++){
        p_sum+=Pij[i];
        if (rnum < p_sum) {
            index = i;
            break;
        }
    }
    
    int i = index_1[index];
    int j = index_2[index];

    return {i, j};
}


//SoftDrop
int SoftDrop(vector <MyClass> finjet, float z_cut, int beta, float R ){
  int fjet = finjet[finjet.size()-1].id;
  //cout <<  "#" << fjet << endl;

  bool done=false;
  while (done==false){
    int ds11 = finjet[fjet-1].d1;
    int ds22 = finjet[fjet-1].d2;
    fastjet::PseudoJet subj1 = finjet[ds11-1].jet;
    fastjet::PseudoJet subj2 = finjet[ds22-1].jet;
    float theta3 = subj1.delta_R(subj2);
    auto [mh3, m03, z3, kt3] = get_m({subj1, subj2});
    float sdcond = z_cut*pow(theta3/R, beta);
    if (z3 > sdcond) {
      //cout << "Done right away" << endl;
      done=true;
    } else{
      //cout << "Not Done" << endl;
      if (subj1.pt() > subj2.pt()){
        if (finjet[ds11-1].d1 == 0 && finjet[ds11-1].d2 == 0) {
          //No children
          done=true;
        }else {
          //cout << "1 > 2" << endl;
          fjet=ds11;
        }
      } else {
        if (finjet[ds22-1].d1 == 0 && finjet[ds22-1].d2 == 0) {
          //No children
          done=true;
        }else {
          //cout << "2 > 1" << endl;
          fjet=ds22;
        }
      }
    }
  }
  //cout << fjet << endl;
  return fjet;
}

auto SoftDrop2(vector <fastjet::PseudoJet> jet, float z_cut, int beta, float R)->std::tuple<vector <fastjet::PseudoJet> , int>{
  vector <fastjet::PseudoJet> finaljet = jet;
  bool done=false;
  int nrGroom = 0;
  while(done==false){
    std::vector <fastjet::PseudoJet > listsubjet = finaljet[0].exclusive_subjets(2);
    float dR = listsubjet[0].delta_R(listsubjet[1]);
    auto [mh3, m03, z3, kt3] = get_m(listsubjet);
    float sdcond = z_cut*pow(dR/R, beta);
    if (z3>sdcond){
      done=true;
    }else{
      //Not done
      if (listsubjet[0].pt() > listsubjet[1].pt()){
        //cout <<10 <<listsubjet[0].constituents().size() << endl;
        
        if (listsubjet[0].has_exclusive_subjets() == false){
          //no children
          done=true;
        }else if (listsubjet[0].exclusive_subjets_up_to(2).size() < 2){
            done=true;
            //cout << "kl" << endl;
        }else{
          //continue with this subjet
          ++nrGroom;
          finaljet={listsubjet[0]};
          //cout << finaljet[0].constituents().size() << endl;
        }
      }else{
        //cout << 20 <<listsubjet[1].constituents().size() << endl;
        if (listsubjet[1].has_exclusive_subjets() == false){
          //no children
          done=true;
        }else if (listsubjet[1].exclusive_subjets_up_to(2).size() < 2){
            done=true;
            //cout << "kl" << endl;
        }else{
          //continue with this subjet
          ++nrGroom;
            finaljet={listsubjet[1]};
            
        }
      }
    }

  }
  return {finaljet, nrGroom};
}

//Make histograms
vector <TH1F*> MakeHists(int bins, float mlargest, float ktlargest, double nlargest){
  TH1F* h1 = new TH1F("h1", "#theta", bins, 0., 1.);
  TH1F* h2 = new TH1F("h2", "m_{hard}^{2}",  bins, 0., mlargest);
  TH1F* h3 = new TH1F("h3", "m_{0}^{2}",  bins, 0., mlargest);
  TH1F* h4 = new TH1F("h4", "z", bins, 0., 0.52);
  TH1F* h5 = new TH1F("h5", "k_{t}", bins, 0., ktlargest);
  TH1F* h6 = new TH1F("h6", "SoftDrop #theta", bins, 0., 1.);
  TH1F* h7 = new TH1F("h7", "SoftDrop z", bins, 0., 0.52);
  TH1F* h8 = new TH1F("h8", "SoftDrop mhard^2",  bins, 0., mlargest);
  TH1F* h9 = new TH1F("h9", "SoftDrop m0^2",  bins, 0., mlargest);
  TH1F* h10 = new TH1F("h10", "N2-N1",  bins, -1, 1);
  return {h1, h2, h3, h4, h5, h6, h7, h8, h9, h10};
}

std::vector <vector <vector <vector <TH1F*>>>> MakeListHist(int bins, float mlargest, float ktlargest, float nlargest){
    //Without background
    std::vector <TH1F*> list_h1 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=1, alpha=1
    std::vector <TH1F*> list_h2 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=1, alpha=0.1
    std::vector <TH1F*> list_h3 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=1, alpha=0.01
    std::vector <vector <TH1F*> > llist_h1= {list_h1, list_h2, list_h3};

    std::vector <TH1F*> list_h4 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=0, alpha=1
    std::vector <TH1F*> list_h5 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=0, alpha=0.1
    std::vector <TH1F*> list_h6 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=0, alpha=0.01
    std::vector <vector <TH1F*> > llist_h2= {list_h4, list_h5, list_h6};

    std::vector <vector <vector <TH1F*> > > lllist_h= {llist_h1, llist_h2}; 

    //With background
    std::vector <TH1F*> list_h21 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=-1, alpha=1
    std::vector <TH1F*> list_h22 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=-1, alpha=0.1
    std::vector <TH1F*> list_h23 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=-1, alpha=0.01
    std::vector <vector <TH1F*> > llist_bg1= {list_h21, list_h22, list_h23};

    std::vector <TH1F*> list_h24 = MakeHists(bins, mlargest, ktlargest,nlargest); //For pow=0, alpha=1
    std::vector <TH1F*> list_h25 = MakeHists(bins, mlargest, ktlargest,nlargest); //For pow=0, alpha=0.1
    std::vector <TH1F*> list_h26 = MakeHists(bins, mlargest, ktlargest,nlargest);; //For pow=0, alpha=0.01
    std::vector <vector <TH1F*> >   llist_bg2= {list_h24, list_h25, list_h26};

    std::vector <vector <vector <TH1F*> > > lllist_bg= {llist_bg1, llist_bg2};


    //With background2
    std::vector <TH1F*> list_h31 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=-1, alpha=1
    std::vector <TH1F*> list_h32 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=-1, alpha=0.1
    std::vector <TH1F*> list_h33 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=-1, alpha=0.01
    std::vector <vector <TH1F*> > llist_bg3= {list_h31, list_h32, list_h33};

    std::vector <TH1F*> list_h34 = MakeHists(bins, mlargest, ktlargest,nlargest); //For pow=0, alpha=1
    std::vector <TH1F*> list_h35 = MakeHists(bins, mlargest, ktlargest,nlargest); //For pow=0, alpha=0.1
    std::vector <TH1F*> list_h36 = MakeHists(bins, mlargest, ktlargest,nlargest);; //For pow=0, alpha=0.01
    std::vector <vector <TH1F*> >   llist_bg4= {list_h34, list_h35, list_h36};

    std::vector <vector <vector <TH1F*> > > lllist_bg2= {llist_bg3, llist_bg4};

    //All togheter
    std::vector <vector <vector <vector <TH1F*>>>> list_all = {lllist_h, lllist_bg, lllist_bg2};

    return list_all;
}

std::vector <vector <vector <TH1F*>>> MakeListHistTrue(int bins, float mlargest, float ktlargest, float nlargest){
    //Without background
    std::vector <TH1F*> list_t1 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=1
    std::vector <TH1F*> list_t2 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=0
    std::vector <vector <TH1F*> > list_true = {list_t1, list_t2};

    //With background
    std::vector <TH1F*> list_t21 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=-1
    std::vector <TH1F*> list_t22 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=0
    std::vector <vector <TH1F*> > list_true_bg = {list_t21, list_t22};

    //With background 2
    std::vector <TH1F*> list_t31 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=-1
    std::vector <TH1F*> list_t32 = MakeHists(bins, mlargest, ktlargest, nlargest); //For pow=0
    std::vector <vector <TH1F*> > list_true_bg2 = {list_t31, list_t32};

    std::vector <vector <vector <TH1F*> >> list_true_all = {list_true, list_true_bg, list_true_bg2 };
    return list_true_all;
}

vector <TH1F*> MakeListHistSpecter(int bins, string name){
  //Without grooming
  TH1F* h2 = new TH1F(Form("h2_%s",name.c_str()), "N_tot  spectrum",  bins, 0., 150.);
  TH1F* h3 = new TH1F(Form("h3_%s",name.c_str()), "#theta  spectrum", bins, 0., 1.);
  TH1F* h4 = new TH1F(Form("h4_%s",name.c_str()), "m  spectrum",      bins, 0., 0.7);
  TH1F* h5 = new TH1F(Form("h5_%s",name.c_str()), "k_t  spectrum",    bins, 0., 0.5);
  TH1F* h6 = new TH1F(Form("h6_%s",name.c_str()), "z",                bins, 0., 0.52);
  TH1F* h7 = new TH1F(Form("h7_%s",name.c_str()), "N_1",              bins, 0., 150.);
  TH1F* h8 = new TH1F(Form("h8_%s",name.c_str()), "N_2",              bins, 0., 150.);
  TH1F* h9 = new TH1F(Form("h9_%s",name.c_str()), "N1-N2/N_tot",      bins, -1.1, 1.1);
  //SoftDrop
  TH1F* h10 = new TH1F(Form("h10_%s",name.c_str()), "SD N_tot  spectrum",  bins, 0., 150.);
  TH1F* h11 = new TH1F(Form("h11_%s",name.c_str()), "SD nr steps  spectrum",  30, 0., 30.);
  TH1F* h12 = new TH1F(Form("h12_%s",name.c_str()), "SD #theta  spectrum", bins, 0., 1.);
  TH1F* h13 = new TH1F(Form("h13_%s",name.c_str()), "SD m  spectrum",      bins, 0., 0.7);
  TH1F* h14 = new TH1F(Form("h14_%s",name.c_str()), "SD k_t  spectrum",    bins, 0., 0.5);
  TH1F* h15 = new TH1F(Form("h15_%s",name.c_str()), "SD z",                bins, 0., 0.52);
  TH1F* h16 = new TH1F(Form("h16_%s",name.c_str()), "SD N_1",              bins, 0., 150.);
  TH1F* h17 = new TH1F(Form("h17_%s",name.c_str()), "SD N_2",              bins, 0., 150.);
  TH1F* h18 = new TH1F(Form("h18_%s",name.c_str()), "SD N1-N2/N_tot",      bins, -1.1, 1.1);
  //Dynamical Grooming
  TH1F* h19 = new TH1F(Form("h19_%s",name.c_str()), "DG N_tot  spectrum",  bins, 0., 150.);
  TH1F* h20 = new TH1F(Form("h20_%s",name.c_str()), "DG #theta  spectrum", bins, 0., 1.);
  TH1F* h21 = new TH1F(Form("h21_%s",name.c_str()), "DG m  spectrum",      bins, 0., 0.7);
  TH1F* h22 = new TH1F(Form("h22_%s",name.c_str()), "DG k_t  spectrum",    bins, 0., 0.5);
  TH1F* h23 = new TH1F(Form("h23_%s",name.c_str()), "DG z",                bins, 0., 0.52);
  TH1F* h24 = new TH1F(Form("h24_%s",name.c_str()), "DG N_1",              bins, 0., 150.);
  TH1F* h25 = new TH1F(Form("h25_%s",name.c_str()), "DG N_2",              bins, 0., 150.);
  TH1F* h26 = new TH1F(Form("h26_%s",name.c_str()), "DG N1-N2/N_tot",      bins, -1.1, 1.1);

  TH1F* h27 = new TH1F(Form("h27_%s",name.c_str()), "After cut N_tot  spectrum",  bins, 0., 150.);
  TH1F* h28 = new TH1F(Form("h28_%s",name.c_str()), "After cut SD nr steps  spectrum",  30, 0., 30.);
  TH1F* h29 = new TH1F(Form("h29_%s",name.c_str()), "After cut #theta  spectrum", bins, 0., 1.);
  TH1F* h30 = new TH1F(Form("h30_%s",name.c_str()), "After cut m  spectrum",      bins, 0., 0.7);
  TH1F* h31 = new TH1F(Form("h31_%s",name.c_str()), "After cut k_t  spectrum",    bins, 0., 0.5);
  TH1F* h32 = new TH1F(Form("h32_%s",name.c_str()), "After cut z",                bins, 0., 0.52);
  TH1F* h33 = new TH1F(Form("h33_%s",name.c_str()), "After cut N_1",              bins, 0., 150.);
  TH1F* h34 = new TH1F(Form("h34_%s",name.c_str()), "After cut N_2",              bins, 0., 150.);
  TH1F* h35 = new TH1F(Form("h35_%s",name.c_str()), "After cut N1-N2/N_tot",      bins, -1.1, 1.1);

  vector <TH1F*> list_hist = {h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32, h33, h34, h35};

  return list_hist;
}

//Plot
int PlotSpectr(string cname, TH1F* h1, int setlog, string axname, string type, string bg){
  TCanvas *c1 = new TCanvas(cname.c_str());
  //h1->SetMarkerStyle(kFullCircle);
  h1->SetStats(0);
  h1->GetXaxis()->SetTitle(axname.c_str());
  h1->GetXaxis()->CenterTitle(true);
  h1->GetYaxis()->SetTitle("#frac{1}{N_{tot}} #frac{dN}{dp_{t}}");
  h1->GetYaxis()->SetTitleOffset(2.0);
  h1->GetYaxis()->CenterTitle(true);
  h1->SetTitle(Form("#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7"));
  h1->Draw("hist");
  if (setlog == 1){
    c1->SetLogy();
    c1->SetLogx();
  }
  c1->SetLeftMargin(0.2);
  gStyle->SetTitleFontSize(0.01);
  c1->Write();
  return 0;
}

int Plot1(std::vector <vector <vector <TH1F*>>> list_true_all, std::vector <vector <vector <vector <TH1F*>>>>list_all, vector <EColor> color, vector <string> labels, vector <string> types, vector <string> wbg, vector <double> alpha_l){
    std::vector <TLegend*> list_l1 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88)};
    std::vector <TLegend*> list_l2 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88)};
    std::vector <TLegend*> list_l3 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88)};
    std::vector <TLegend*> list_l4 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88)};

    std::vector <vector <TLegend*> > list_l = {list_l1, list_l2};
    std::vector <vector <TLegend*> > list_lbg = {list_l3, list_l4};
    std::vector <vector <vector <TLegend*>>> list_l_all = {list_l, list_lbg};

  
    vector <string> list_c1 = {"c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"};
    vector <string> list_c2 = {"c1_2", "c2_2", "c3_2", "c4_2", "c5_2", "c6_2", "c7_2", "c8_2", "c9_2", "c10_2"};
    vector <string> list_c3 = {"c1_3", "c2_3", "c3_3", "c4_3", "c5_3", "c6_3", "c7_3", "c8_3", "c9_3", "c10_3"};
    vector <string> list_c4 = {"c1_4", "c2_4", "c3_4", "c4_4", "c5_4", "c6_4", "c7_4", "c8_4", "c9_4", "c10_4"};

    vector <vector <string> > list_c = {list_c1, list_c2};
    vector <vector <string> > list_cbg = {list_c3, list_c4};
    vector <vector <vector <string> > > list_c_all = {list_c, list_cbg};

    TCanvas* c ;
    for (int m=0; m<list_all.size(); ++m){
        for (int i=0; i<list_all[0].size(); ++i){
            for (int j=0; j<list_all[0][0][0].size(); ++j){
                c = new TCanvas(list_c_all[m][i][j].c_str());
                list_true_all[m][i][j]->SetLineColor(kGray);
                list_true_all[m][i][j]->SetFillColor(kGray);
                list_true_all[m][i][j]->SetLineWidth(3);
                list_true_all[m][i][j]->GetYaxis()->SetTitle("arb. unit");
                list_true_all[m][i][j]->GetXaxis()->SetTitle(labels[j].c_str());
                list_true_all[m][i][j]->SetTitle(Form("%s %s distribution, %s", types[i].c_str(),labels[j].c_str(), wbg[m].c_str()));
                list_true_all[m][i][j]->Draw("hist");
                list_true_all[m][i][j]->SetStats(0);
                list_l_all[m][i][j]->AddEntry(list_true_all[m][i][j], "Classic value");
                for (int k=0; k<list_all[0][0].size(); ++k){
                list_all[m][i][k][j]->SetLineStyle(kDashed);
                list_all[m][i][k][j]->SetLineColor(color[k]);
                list_all[m][i][k][j]->SetLineWidth(3);    
                list_all[m][i][k][j]->Draw("hist same");
                list_all[m][i][k][j]->SetStats(0);
                list_l_all[m][i][j]->AddEntry(list_all[m][i][k][j], Form("#alpha = %.2f", alpha_l[k]));
                }
                list_l_all[m][i][j]->SetBorderSize(0);
                list_l_all[m][i][j]->SetFillColor(0);
                list_l_all[m][i][j]->Draw();
                c->Write();
            }
        
        }
    }

    return 0;
}


int Plot2(std::vector <vector <vector <TH1F*>>> list_true_all, std::vector <vector <vector <vector <TH1F*>>>>list_all, vector <EColor> color, vector <string> labels, vector <string> types, vector <string> wbg, vector <double> alpha_l){

    vector <string> list_c_2 = {"theta_1", "z_1", "mh_1", "m0_1", "N_1"};
    vector <string> list_c_3 = {"theta_2", "z_2", "mh_2", "m0_2", "N_2"};
    vector <string> list_c_4 = {"theta_3", "z_3", "mh_3", "m0_3", "N_3"};
    vector <string> list_c_5 = {"theta_wbg_1", "z_wbg_1", "mh_wbg_1", "m0_wbg_1", "N_wbg_1"};
    vector <string> list_c_6 = {"theta_wbg_2", "z_wbg_2", "mh_wbh_2", "m0_wbg_2", "N_wbg_2"};
    vector <string> list_c_7 = {"theta_wbg_3", "z_wbg_3", "mh_wbg_3", "m0_wbg_3", "N_wbg_3"};
    vector <string> list_c_8 = {"theta_wbg2_1", "z_wbg2_1", "mh_wbg2_1", "m0_wbg2_1", "N_wbg2_1"};
    vector <string> list_c_9 = {"theta_wbg2_2", "z_wbg2_2", "mh_wbh2_2", "m0_wbg2_2", "N_wbg2_2"};
    vector <string> list_c_10 = {"theta_wbg2_3", "z_wbg2_3", "mh_wbg2_3", "m0_wbg2_3", "N_wbg2_3"};
    vector <vector <string>> list_c_k = {list_c_2, list_c_3, list_c_4};
    vector <vector <string>> list_c_l = {list_c_5, list_c_6, list_c_7};
    vector <vector <string>> list_c_m = {list_c_8, list_c_9, list_c_10};
    vector <vector <vector <string>>> list_c_double = {list_c_k, list_c_l, list_c_m};


    vector <TLegend*> list_l_2 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};
    vector <TLegend*> list_l_3 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};
    vector <TLegend*> list_l_4 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};
    vector <TLegend*> list_l_5 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};
    vector <TLegend*> list_l_6 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};
    vector <TLegend*> list_l_7 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};
    vector <TLegend*> list_l_8 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};
    vector <TLegend*> list_l_9 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};
    vector <TLegend*> list_l_10 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};

    vector <vector <TLegend*>> list_l_k = {list_l_2, list_l_3, list_l_4};
    vector <vector <TLegend*>> list_l_l = {list_l_5, list_l_6, list_l_7};
    vector <vector <TLegend*>> list_l_m = {list_l_8, list_l_9, list_l_10};

    vector <vector <vector <TLegend*>>> list_l_double = {list_l_k, list_l_l, list_l_m};

    TCanvas* c ;
    for (int m=0; m<list_all.size(); ++m){
        for (int i=5; i<labels.size(); i++){
            for (int pow=0; pow<list_all[0][0].size(); ++pow){
                c=new TCanvas(list_c_double[m][pow][i-5].c_str());
                list_true_all[m][0][i]->SetLineColor(kRed); //Anti-kt m_h
                list_true_all[m][0][i]->SetFillColor(kRed);
                list_true_all[m][1][i]->SetLineColor(kBlack); //C/A m_h
                list_true_all[m][1][i]->SetFillColor(kBlack);
                list_true_all[m][0][i]->GetYaxis()->SetTitle("arb. unit");
                list_true_all[m][0][i]->GetXaxis()->SetTitle(labels[i].c_str());
                list_true_all[m][0][i]->SetTitle(Form("%s distribution, %s ", labels[i].c_str(), wbg[m].c_str()));
                list_true_all[m][0][i]->Draw("hist");
                list_true_all[m][1][i]->Draw("hist same");
                list_true_all[m][0][i]->SetStats(0);
                list_true_all[m][1][i]->SetStats(0);
                list_l_double[m][pow][i-5]->AddEntry(list_true_all[m][0][i], Form("Classic value %s", types[0].c_str()));
                list_l_double[m][pow][i-5]->AddEntry(list_true_all[m][1][i], Form("Classic value %s", types[1].c_str()));
                list_all[m][0][pow][i]->SetLineColor(kRed); //Anti-kt m_h alpha=1
                list_all[m][0][pow][i]->SetLineStyle(kDashed);
                list_all[m][0][pow][i]->Draw("hist same");
                list_all[m][0][pow][i]->SetStats(0);
                list_l_double[m][pow][i-5]->AddEntry(list_all[m][0][pow][i], Form("#alpha = %.2f with %s", alpha_l[pow], types[0].c_str()));
                list_all[m][1][pow][i]->SetLineColor(kBlack); //C/A m_h alpha=1
                list_all[m][1][pow][i]->SetLineStyle(kDashed);
                list_all[m][1][pow][i]->Draw("hist same");
                list_all[m][1][pow][i]->SetStats(0);
                list_l_double[m][pow][i-5]->AddEntry(list_all[m][1][pow][i], Form("#alpha = %.2f with %s", alpha_l[pow], types[1].c_str()));
                list_l_double[m][pow][i-5]->SetFillColor(0);
                list_l_double[m][pow][i-5]->SetBorderSize(0);
                list_l_double[m][pow][i-5]->Draw();
                c->Write();
            }
        }
    } 
    return 0;
}


int Plot3(std::vector <vector <vector <TH1F*>>> list_true_all, std::vector <vector <vector <vector <TH1F*>>>>list_all, vector <EColor> color, vector <string> labels, vector <string> types, vector <string> wbg, vector <double> alpha_l){
    vector <string> list_c_2 = {"theta_kt_1", "z_kt_1", "mh_kt_1", "m0_kt_1", "N_kt_1"};
    vector <string> list_c_3 = {"theta_kt_2", "z_kt_2", "mh_kt_2", "m0_kt_2", "N_kt_2"};
    vector <string> list_c_4 = {"theta_kt_3", "z_kt_3", "mh_kt_3", "m0_kt_3", "N_kt_3"};

    vector <string> list_c_5 = {"theta_CA_1", "z_CA_1", "mh_CA_1", "m0_CA_1", "N_CA_1"};
    vector <string> list_c_6 = {"theta_CA_2", "z_CA_2", "mh_wbh_2", "m0_CA_2", "N_CA_2"};
    vector <string> list_c_7 = {"theta_CA_3", "z_CA_3", "mh_CA_3", "m0_CA_3", "N_CA_3"};
    
    vector <vector <string>> list_c_k = {list_c_2, list_c_3, list_c_4};
    vector <vector <string>> list_c_l = {list_c_5, list_c_6, list_c_7};
    vector <vector <vector <string>>> list_c_double = {list_c_k, list_c_l};


    vector <TLegend*> list_l_2 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};
    vector <TLegend*> list_l_3 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};
    vector <TLegend*> list_l_4 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};

    vector <TLegend*> list_l_5 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};
    vector <TLegend*> list_l_6 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};
    vector <TLegend*> list_l_7 = {new TLegend(0.3, 0.8, 0.15, 0.88), new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88),new TLegend(0.3, 0.8, 0.15, 0.88)};

    vector <vector <TLegend*>> list_l_k = {list_l_2, list_l_3, list_l_4};
    vector <vector <TLegend*>> list_l_l = {list_l_5, list_l_6, list_l_7};

    vector <vector <vector <TLegend*>>> list_l_double = {list_l_k, list_l_l};

    TCanvas* c ;
    for (int typ=0; typ<2; ++typ){
        for (int i=5; i<labels.size(); ++i){
            for (int pow=0; pow<3; ++pow){
                c = new TCanvas(list_c_double[typ][pow][i-5].c_str());
                list_true_all[0][typ][i]->SetLineColor(kRed);
                list_true_all[0][typ][i]->SetFillColor(kRed);
                list_true_all[1][typ][i]->SetLineColor(kBlack);
                list_true_all[1][typ][i]->SetFillColor(kBlack);
                list_true_all[2][typ][i]->SetLineColor(kBlue);
                list_true_all[2][typ][i]->SetFillColor(kBlue);
                list_true_all[0][typ][i]->GetYaxis()->SetTitle("arb. unit");
                list_true_all[0][typ][i]->GetXaxis()->SetTitle(labels[i].c_str());
                list_true_all[0][typ][i]->SetTitle(Form("%s distribution, for %s with alpha %.2f", labels[i].c_str(), types[typ].c_str(), alpha_l[pow]));
                list_true_all[0][typ][i]->Draw("hist");
                list_true_all[1][typ][i]->Draw("hist same");
                list_true_all[2][typ][i]->Draw("hist same");
                list_true_all[2][typ][i]->SetStats(0);
                list_true_all[0][typ][i]->SetStats(0);
                list_true_all[1][typ][i]->SetStats(0);
                list_all[0][typ][pow][i]->SetLineColor(kRed); //without bg, Anti-kt, alpha=1, mh
                list_all[0][typ][pow][i]->SetLineStyle(kDashed);
                list_all[0][typ][pow][i]->SetLineWidth(3);
                list_all[0][typ][pow][i]->Draw("hist same");
                list_all[0][typ][pow][i]->SetStats(0);
                list_all[1][typ][pow][i]->SetLineColor(kBlack); //with bg, Anti-kt, alpha=1, mh
                list_all[1][typ][pow][i]->SetLineStyle(kDashed);
                list_all[1][typ][pow][i]->SetLineWidth(3);
                list_all[1][typ][pow][i]->Draw("hist same");
                list_all[1][typ][pow][i]->SetStats(0);
                list_all[2][typ][pow][i]->SetLineColor(kBlue); //with bg, Anti-kt, alpha=1, mh
                list_all[2][typ][pow][i]->SetLineStyle(kDashed);
                list_all[2][typ][pow][i]->SetLineWidth(3);
                list_all[2][typ][pow][i]->Draw("hist same");
                list_all[2][typ][pow][i]->SetStats(0);
                list_l_double[typ][pow][i-5]->AddEntry(list_true_all[0][typ][i], Form("Classic value %s", wbg[0].c_str()));
                list_l_double[typ][pow][i-5]->AddEntry(list_true_all[1][typ][i], Form("Classic value %s", wbg[1].c_str()));
                list_l_double[typ][pow][i-5]->AddEntry(list_true_all[2][typ][i], Form("Classic value %s", wbg[2].c_str()));
                list_l_double[typ][pow][i-5]->AddEntry(list_all[0][typ][pow][i], Form("%s ", wbg[0].c_str()));
                list_l_double[typ][pow][i-5]->AddEntry(list_all[1][typ][pow][i], Form("%s ", wbg[1].c_str()));
                list_l_double[typ][pow][i-5]->AddEntry(list_all[2][typ][pow][i], Form("%s ", wbg[2].c_str()));
                list_l_double[typ][pow][i-5]->SetFillColor(0);
                list_l_double[typ][pow][i-5]->SetBorderSize(0);
                list_l_double[typ][pow][i-5]->Draw();
                c->Write();
            }
        }
    }
    return 0;
}


int PlotSpectr2(string cname, TH1F* h1, TH1F* h2, TH1F*h3, int bins, string axname, string type, vector <string> bg){
  TCanvas *c1 = new TCanvas(cname.c_str());
  TLegend *l1 = new TLegend(0.89, 0.81, 0.59, 0.88);
  string extra = "";

    
    h1->Scale(1./h1->Integral());
    h1->GetYaxis()->SetTitle(Form("#frac{1}{N_{jet}} #frac{dN}{d%s}",axname.c_str()));
    h1->GetYaxis() -> SetTitleOffset(2.0);
    h1->GetXaxis()->SetTitle(axname.c_str());
    h1->GetYaxis()->CenterTitle(true);
    h1->GetXaxis()->CenterTitle(true);
    h2->Scale(1./h2->Integral());
    h2->GetYaxis()->SetTitle(Form("#frac{1}{N_{jet}} #frac{dN}{d%s}",axname.c_str()));
    h2->GetYaxis() -> SetTitleOffset(2.0);
    h2->GetXaxis()->SetTitle(axname.c_str());
    h2->GetYaxis()->CenterTitle(true);
    h2->GetXaxis()->CenterTitle(true);
    
    h3->Scale(1./h3->Integral());
    h3->GetYaxis()->SetTitle(Form("#frac{1}{N_{jet}} #frac{dN}{d%s}",axname.c_str()));
    h3->GetYaxis()-> SetTitleOffset(2.0);
    h3->GetXaxis()->SetTitle(axname.c_str());
    h3->GetYaxis()->CenterTitle(true);
    h3->GetXaxis()->CenterTitle(true);
    

    if (cname.find('g') != std::string::npos){
        h1->GetYaxis()->SetRangeUser(pow(10,-6), 10.);
        h2->GetYaxis()->SetRangeUser(pow(10,-6), 10.);
        
        h3->GetYaxis()->SetRangeUser(pow(10,-6), 10.);
        
    }

  h1->SetLineColor(kRed);
  h1->SetLineStyle(kDashed);
  h1->SetLineWidth(3);
  h2->SetLineColor(kBlack);
  h2->SetLineStyle(kDashed);
  h2->SetLineWidth(3);
  
  h3->SetLineColor(kBlue);
  h3->SetLineStyle(kDashed);
  h3->SetLineWidth(3);
  
  h1->SetStats(0);
  h2->SetStats(0);
  h3->SetStats(0);

    if (cname.find('S') != std::string::npos){
        extra = "SoftDrop z_{cut}=0.1, #beta=1";
        h1->SetTitle(Form("#splitline{#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7, reclustered with %s}{%s}", type.c_str(), extra.c_str()));
        h2->SetTitle(Form("#splitline{#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7, reclustered with %s}{%s}", type.c_str(), extra.c_str()));
        
        h3->SetTitle(Form("#splitline{#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7, reclustered with %s}{%s}", type.c_str(), extra.c_str()));
    } else if (cname.find('D') != std::string::npos){
        extra = "Dynamical Grooming a=1";
        h1->SetTitle(Form("#splitline{#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7, reclustered with %s}{%s}", type.c_str(), extra.c_str()));
        h2->SetTitle(Form("#splitline{#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7, reclustered with %s}{%s}", type.c_str(), extra.c_str()));
        h3->SetTitle(Form("#splitline{#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7, reclustered with %s}{%s}", type.c_str(), extra.c_str()));
    }else if (cname.find('c') != std::string::npos){
        extra = "SoftDrop z_{cut}=0.1, #beta=1, #frac{N_{1}-N_{2}}{N_{tot}} < - 0.5";
        h1->SetTitle(Form("#splitline{#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7, reclustered with %s}{%s}", type.c_str(), extra.c_str()));
        h2->SetTitle(Form("#splitline{#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7, reclustered with %s}{%s}", type.c_str(), extra.c_str()));
        h3->SetTitle(Form("#splitline{#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7, reclustered with %s}{%s}", type.c_str(), extra.c_str()));
    }else {
        h1->SetTitle(Form("#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7, reclustered with %s", type.c_str()));
        h2->SetTitle(Form("#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7, reclustered with %s", type.c_str()));
        h3->SetTitle(Form("#hat{p_{t,}}_{min}=200GeV, anti-k_{t}, R=0.7, reclustered with %s", type.c_str()));
    }
    

  //Decide what is drawn first 
  vector <float> list_content_1;
  vector <float> list_content_2;
  vector <float> list_content_3;
  for (int i=0; i<bins; ++i){
    list_content_1.push_back( h1->GetBinContent(i));
    list_content_2.push_back( h2->GetBinContent(i));
    list_content_3.push_back( h3->GetBinContent(i));
  }
  
  if (*max_element(list_content_1.begin(), list_content_1.end()) > *max_element(list_content_2.begin(), list_content_2.end()) && *max_element(list_content_1.begin(), list_content_1.end())> *max_element(list_content_3.begin(), list_content_3.end())){
    h1->Draw("hist");
    h2->Draw("hist same");
    h3->Draw("hist same");
  } else if(*max_element(list_content_2.begin(), list_content_2.end()) > *max_element(list_content_1.begin(), list_content_1.end()) && *max_element(list_content_2.begin(), list_content_2.end())> *max_element(list_content_3.begin(), list_content_3.end())){
    h2->Draw("hist");
    h1->Draw("hist same");
    h3->Draw("hist same");
  } else {
    h3->Draw("hist");
    h2->Draw("hist same");
    h1->Draw("hist same");
  }
    gStyle->SetTitleFontSize(0.01);
  l1->AddEntry(h1, Form("%s", bg[0].c_str()));
  l1->AddEntry(h2, Form("%s", bg[1].c_str()));
  l1->AddEntry(h3, Form("%s", bg[2].c_str()));
  l1->SetFillColor(0);
  l1->SetBorderSize(0);
  l1->SetTextSize(6);
  l1->Draw();
  c1->SetLeftMargin(0.2);
  c1->Write();

  return 0;
}


//Lund Plane

int LundPlane(fastjet::PseudoJet jet, TH2F* h2D, float w, float R){
  vector <fastjet::PseudoJet> currentlist = jet.exclusive_subjets(2);
  float theta3, theta4;
  bool done = false;
  int nrdeclust = 0;
  while (done==false){
    ++nrdeclust;
    theta3 = currentlist[0].delta_R(currentlist[1]);
    auto [mh3, m03, z3, kt3] = get_m(currentlist);
    h2D->Fill(log(R/theta3), log(sqrt(kt3)/jet.pt()),  w);
    if (currentlist[0].pt() > currentlist[1].pt()){
      if (currentlist[0].exclusive_subjets_up_to(2).size()<2) {
        done=true;
        continue;
      }
      //continue with this jet
      currentlist = currentlist[0].exclusive_subjets(2);
    }
    if (currentlist[1].pt() > currentlist[0].pt()){
      if (currentlist[0].exclusive_subjets_up_to(2).size()<2) {
        done = true;
        continue;
      }
      //continue with this jet
      currentlist = currentlist[1].exclusive_subjets(2);
    }
  }
  //h11->Fill(nrdeclust, w);
  return 0;
}


int LundPlaneQ(vector <MyClass> Qjet, TH2F* h2D, float w, float R){
    //Extract information from the object
    int id2 = Qjet[Qjet.size()-1].id;
    int ds1 = Qjet[id2-1].d1;
    int ds2 = Qjet[id2-1].d2;

    //Save the two jets at the hardest splitting
    vector <fastjet::PseudoJet> listjet1;
    listjet1.push_back(Qjet[ds1-1].jet);
    listjet1.push_back(Qjet[ds2-1].jet);

    vector <fastjet::PseudoJet> currentlist = listjet1;
    vector <int> currentids = {ds1, ds2};
    vector <fastjet::PseudoJet> secondlist;
    float theta3, theta4;
    bool done = false;
    int nrdeclust = 0;
    while (done==false){
        ++nrdeclust;
        theta3 = currentlist[0].delta_R(currentlist[1]);
        auto [mh3, m03, z3, kt3] = get_m(currentlist);
        h2D->Fill(log(R/theta3), log(sqrt(kt3)/Qjet[id2].jet.pt()),  w);

        if (currentlist[0].pt() > currentlist[1].pt()){
            if (Qjet[currentids[0]-1].nrparticles < 2) {
                done=true;

                continue;
            }
            //continue with this jet
            currentlist = {Qjet[Qjet[currentids[0]-1].d1-1].jet, Qjet[Qjet[currentids[0]-1].d2-1].jet };
            currentids = {Qjet[currentids[0]-1].d1, Qjet[currentids[0]-1].d2};
        } else {
            if (Qjet[currentids[1]-1].nrparticles < 2) {
                done = true;
                continue;
            }
            //continue with this jet
            currentlist = {Qjet[Qjet[currentids[1]-1].d1-1].jet, Qjet[Qjet[currentids[1]-1].d2-1].jet };
            currentids = {Qjet[currentids[1]-1].d1, Qjet[currentids[1]-1].d2};
        }
    }
    //h11->Fill(nrdeclust, w);
    return 0;
}


//Constituent subtraction
auto GetRho(vector <fastjet::PseudoJet> sortedJets)->std::tuple<float, float>{
    vector <float> list_median_pt;
    vector <float> list_median_m;
    for (int k=0; k<sortedJets.size(); ++k){
        std::vector<fastjet::PseudoJet> particles, ghosts;
        fastjet::SelectorIsPureGhost().sift(sortedJets[k].constituents(), ghosts, particles);

        vector <float> pt;
        vector <float> m;
        for (int i=0; i<particles.size(); ++i){
            pt.push_back(particles[i].pt());
            m.push_back(mdelta(particles[i]));
        }
        float sum_pt = std::accumulate(pt.begin(), pt.end(), 0.0f);
        float sum_m = std::accumulate(m.begin(), m.end(), 0.0f);
        
        list_median_pt.push_back(sum_pt/sortedJets[k].area());
        list_median_m.push_back(sum_m/sortedJets[k].area());
    }

    sort(list_median_pt.begin(), list_median_pt.end());
    sort(list_median_m.begin(), list_median_m.end());
    float rho, rho_m;
    if (sortedJets.size()%2){
        rho = (list_median_pt[sortedJets.size()/2]+list_median_pt[sortedJets.size()/2 +1])/2;
        rho_m = (list_median_m[sortedJets.size()/2]+list_median_m[sortedJets.size()/2 +1])/2;
    } else{
        rho = list_median_pt[ceil(sortedJets.size()/2)];
        rho_m = list_median_m[ceil(sortedJets.size()/2)];
    }
    return {rho, rho_m};
}

vector <fastjet::PseudoJet> ConstituentSubtraction(fastjet::PseudoJet jet, float rho, float rho_m){
    std::vector<fastjet::PseudoJet> particles, ghosts;
    fastjet::SelectorIsPureGhost().sift(jet.constituents(), ghosts, particles);
    
    vector<fastjet::PseudoJet> selected_particles=particles;
    
    //Make ghost particles
    double ghosts_area, ptg, mg, phi, y;
    vector <fastjet::PseudoJet> ghost_particles;
    unsigned long nGhosts=ghosts.size();
    for (unsigned int j=0;j<nGhosts; ++j){
        ghosts_area=ghosts[j].area();
        ptg = ghosts_area*rho;
        mg = ghosts_area*rho_m;
        phi = ghosts[j].phi();
        y = ghosts[j].eta();
        ghost_particles.push_back(fastjet::PseudoJet(ptg*cos(phi), ptg*sin(phi), (ptg+mg)*sinh(y), (ptg+mg)*cosh(y)));
    }

    
    //Do subtraction
    vector <double> dR;
    vector <int> index1, index2;
    vector <float> pti, ptk, mdi, mdk;
   
    for (int i = 0; i<particles.size(); ++i){
        pti.push_back(particles[i].pt());
        mdi.push_back(mdelta(particles[i]));
        for (int k=0; k<ghost_particles.size(); ++k){
            dR.push_back(particles[i].delta_R(ghost_particles[k]));
            if (ptk.size() < ghost_particles.size()){
                ptk.push_back(ghost_particles[k].pt());
                mdk.push_back(mdelta(ghost_particles[k]));
            }
            index1.push_back(i);
            index2.push_back(k);
        }
    }

    vector<size_t> idx(dR.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values 
    stable_sort(idx.begin(), idx.end(), [&dR](size_t i1, size_t i2) {return dR[i1] < dR[i2];});
    int i1, i2;    
    for (int i=0; i<dR.size(); ++i){
        //starting from smallest dR
        i1 = index1[idx[i]];
        i2 = index2[idx[i]];

        if (particles[i1].pt()>=ghost_particles[i2].pt()){
            pti[i1] = particles[i1].pt()-ghost_particles[i2].pt();
            ptk[i2] = 0.;
        } else{
            pti[i1]=0.;
            ptk[i2] = ghost_particles[i2].pt()-particles[i1].pt();
        }
        if (mdelta(particles[i1])>=mdelta(ghost_particles[i2])){
            mdi[i1] = mdelta(particles[i1])-mdelta(ghost_particles[i2]);
            mdk[i2] = 0.;
        } else{
            mdi[i1]=0.;
            mdk[i2] = mdelta(ghost_particles[i2])-mdelta(particles[i1]);
        }
    }

    vector <fastjet::PseudoJet> surviving_particles;
    for (int i=0; i<particles.size(); ++i){
        if (pti[i]== 0.0) continue;
        surviving_particles.push_back(fastjet::PseudoJet(pti[i]*cos(particles[i].phi()), pti[i]*sin(particles[i].phi()), (pti[i]+mdi[i])*sinh(particles[i].eta()), (pti[i]+mdi[i])*cosh(particles[i].eta())));
    }

    if (surviving_particles.size()==0) {
        cout << "oh no" << endl;
    }

    return surviving_particles;    
    
}


vector <fastjet::PseudoJet> GetParticleList(vector <fastjet::PseudoJet> Inputs, vector <fastjet::PseudoJet> bgparticles, fastjet::JetDefinition jetDef, fastjet::AreaDefinition areaDef, double pTMin, float R, int index=0, fastjet::PseudoJet medjet = fastjet::PseudoJet(0,0,0,0)){
    vector <fastjet::PseudoJet> allInputs = Inputs;
    allInputs.insert(allInputs.end(), bgparticles.begin(), bgparticles.end());

    vector <fastjet::PseudoJet> inclusiveJets, sortedJets;
    fastjet::ClusterSequenceArea clustSeq(allInputs, jetDef, areaDef);
    inclusiveJets = clustSeq.inclusive_jets(pTMin);
    sortedJets    = sorted_by_pt(inclusiveJets);
    //cout << sortedJets[index].constituents().size() << endl;
    //cout << sortedJets[index].delta_R(medjet) << endl;
    if (sortedJets[index].delta_R(medjet) > 0.4){
      double dRlim = 0.4;
      double mindR = 1.;
      int nr = 0;
      int same = 0;
      while (mindR>dRlim){
        if (nr == sortedJets.size()-1){
          break;
        }
        double dR = medjet.delta_R(sortedJets[nr]);
        if (dR < mindR){
          same = nr;
          mindR = dR;
        } else {
          ++nr;
        }
      }

      index = same;
    }
    
    fastjet::JetDefinition jet_def_for_rho(fastjet::kt_algorithm, 0.4);
    fastjet::Selector rho_range =  fastjet::SelectorAbsRapMax(3.0);
    //fastjet::ClusterSequenceArea clust_seq_rho(allInputs, jetDef, areaDef);  

    fastjet::JetMedianBackgroundEstimator bge_rho(rho_range, jet_def_for_rho, areaDef);

    fastjet::BackgroundJetScalarPtDensity *scalarPtDensity=new fastjet::BackgroundJetScalarPtDensity();
    bge_rho.set_jet_density_class(scalarPtDensity);
    bge_rho.set_particles(allInputs);

    fastjet::contrib::ConstituentSubtractor subtractor(&bge_rho);

    fastjet::PseudoJet subtracted_jet = subtractor(sortedJets[index]);

    //cout << subtracted_jet.constituents().size() << endl;

    //auto [rho, rho_m] = GetRho(sortedJets);
    //vector <fastjet::PseudoJet> surviving_particles = ConstituentSubtraction(sortedJets[index], rho, rho_m);
    //cout << surviving_particles.size() << endl;
    return subtracted_jet.constituents();
}

auto GetParticleList2(vector <fastjet::PseudoJet> Inputs, vector <fastjet::PseudoJet> bgparticles, fastjet::JetDefinition jetDef, fastjet::AreaDefinition areaDef, double pTMin, float R, int index=0, fastjet::PseudoJet medjet = fastjet::PseudoJet(0,0,0,0))->std::tuple<int, int>{
    vector <fastjet::PseudoJet> allInputs = Inputs;
    allInputs.insert(allInputs.end(), bgparticles.begin(), bgparticles.end());

    vector <fastjet::PseudoJet> inclusiveJets, sortedJets;
    fastjet::ClusterSequenceArea clustSeq(allInputs, jetDef, areaDef);
    inclusiveJets = clustSeq.inclusive_jets(pTMin);
    sortedJets    = sorted_by_pt(inclusiveJets);
    //cout << sortedJets[index].constituents().size() << endl;
    //cout << sortedJets[index].delta_R(medjet) << endl;
    if (sortedJets[index].delta_R(medjet) > 0.4){
      double dRlim = 0.4;
      double mindR = 1.;
      int nr = 0;
      int same = 0;
      while (mindR>dRlim){
        if (nr == sortedJets.size()-1){
          //cout << "too high" <<  mindR<<endl;
          //if (mindR >1){
          //  cout << "not ok" << endl;
          //}
          break;
        }
        double dR = medjet.delta_R(sortedJets[nr]);
        //cout << dR << endl;
        if (dR < mindR){
          same = nr;
          mindR = dR;
        } else {
          ++nr;
        }
      }
      //cout << "new" << sortedJets[nr].delta_R(medjet) << "old" << sortedJets[index].delta_R(medjet)<< endl;
      //cout << nr << endl;
      index = nr;
    }
    
    fastjet::JetDefinition jet_def_for_rho(fastjet::kt_algorithm, 0.4);
    fastjet::Selector rho_range =  fastjet::SelectorAbsRapMax(3.0);
    //fastjet::ClusterSequenceArea clust_seq_rho(allInputs, jetDef, areaDef);  

    fastjet::JetMedianBackgroundEstimator bge_rho(rho_range, jet_def_for_rho, areaDef);

    fastjet::BackgroundJetScalarPtDensity *scalarPtDensity=new fastjet::BackgroundJetScalarPtDensity();
    bge_rho.set_jet_density_class(scalarPtDensity);
    bge_rho.set_particles(allInputs);

    fastjet::contrib::ConstituentSubtractor subtractor(&bge_rho);

    fastjet::PseudoJet subtracted_jet = subtractor(sortedJets[index]);

    //cout << "#" << subtracted_jet.constituents().size() << endl;

    auto [rho, rho_m] = GetRho(sortedJets);
    vector <fastjet::PseudoJet> surviving_particles = ConstituentSubtraction(sortedJets[index], rho, rho_m);
    //cout << surviving_particles.size()<< endl;
    return {subtracted_jet.constituents().size(), surviving_particles.size()};
}

/*
vector <fastjet::PseudoJet> GetParticleList2(vector <fastjet::PseudoJet> Inputs, vector <fastjet::PseudoJet> bgparticles, fastjet::JetDefinition jetDef, fastjet::AreaDefinition areaDef, double pTMin, float R){

    vector <fastjet::PseudoJet> allInputs = Inputs;
    allInputs.insert(allInputs.end(), bgparticles.begin(), bgparticles.end());

    vector <fastjet::PseudoJet> inclusiveJets, sortedJets;
    fastjet::ClusterSequenceArea clustSeq(allInputs, jetDef, areaDef);
    inclusiveJets = clustSeq.inclusive_jets(pTMin);
    sortedJets    = sorted_by_pt(inclusiveJets);

    fastjet::JetDefinition jet_def_for_rho(fastjet::kt_algorithm, 0.7);
    fastjet::Selector rho_range =  fastjet::SelectorAbsRapMax(2.5);

    fastjet::ClusterSequenceArea clust_seq_rho(allInputs, jetDef, areaDef);  

    fastjet::JetMedianBackgroundEstimator bge_rho(rho_range, jet_def_for_rho, areaDef);
    fastjet::BackgroundJetScalarPtDensity *scalarPtDensity=new fastjet::BackgroundJetScalarPtDensity();
    bge_rho.set_jet_density_class(scalarPtDensity); 
    bge_rho.set_particles(allInputs);

    // subtractor:
    //fastjet::contrib::ConstituentSubtractor subtractor(&bge_rho);  

    //fastjet::PseudoJet subtracted_jet = subtractor(sortedJets[0]);



    return sortedJets[0].constituents();
}
*/

//Dynamical Grooming
fastjet::PseudoJet dynamicalGrooming(fastjet::PseudoJet jet, float R, float a){

    vector <float> hardness;
    vector <fastjet::PseudoJet> hardjet;
    fastjet::PseudoJet currentjet = jet;
    vector <fastjet::PseudoJet> currentlist;


    float theta3;
    bool done = false;
    while (done==false){

        currentlist = currentjet.exclusive_subjets(2);
        theta3 = currentlist[0].delta_R(currentlist[1]);
        auto [mh3, m03, z3, kt3] = get_m(currentlist);
        hardness.push_back(z3*(1-z3)*currentjet.pt()*pow(theta3/R, a));
        hardjet.push_back(currentjet);
        if (currentlist[0].pt() > currentlist[1].pt()){
            if (currentlist[0].exclusive_subjets_up_to(2).size()<2) {
                done=true;
                continue;
            }
            currentjet = currentlist[0];
        } else {
            if (currentlist[1].exclusive_subjets_up_to(2).size()<2) {
                done=true;
                continue;
            }
            currentjet = currentlist[1];
        }
    }
    
    vector<size_t> idx(hardness.size());
    iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(), [&hardness](size_t i1, size_t i2) {return hardness[i1] > hardness[i2];});
   
    fastjet::PseudoJet hardest_split = hardjet[idx[0]];

    return hardest_split;
}

