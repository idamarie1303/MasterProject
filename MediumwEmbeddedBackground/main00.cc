// My own


#include "Pythia8/Pythia.h" 

// This is the minimal interface needed to access FastJet.
// A more sophisticated interface is demonstrated in main72.cc.
#include "Pythia8Plugins/FastJet3.h"
#include "fastjet/ClusterSequenceArea.hh"
#include "fjcontrib/RecursiveTools/SoftDrop.hh"
#include "TH1F.h"
#include "TFile.h" 
#include "TH2F.h" 
#include "TCanvas.h"
#include "main72.h"  
#include "TPad.h"
#include "nlohmann/json.hpp"

#include <fstream>

#include <chrono>
#include <iostream>
#include <fstream>

using namespace Pythia8;
using json = nlohmann::json;

void tokenize(std::string const &str, const char* delim,
            std::vector<std::string> &out) //hey
{
    char *token = strtok(const_cast<char*>(str.c_str()), delim);
    while (token != nullptr)
    {
        out.push_back(std::string(token));
        token = strtok(nullptr, delim);
    }
}


struct Point { double E, x, y, z; };


void to_json(json& j, const Point& p)
{
    j = {{"E", p.E},{"px", p.x}, {"py", p.y}, {"pz", p.z}};
}


int main() {
  std::ofstream o1("hadron_0bkg.json");
  std::ofstream o2("hadron_0bkgsub.json");
  


  cout << "start" << endl;
  fstream my_file;
  const char* delim = " ";
  vector <std::string> lines;
  string str = "event";
  string str2 = "end";
  string str3 = "weight";
  string lastevent = "# event 0";
  int num_lines = 0;
  string line;
  double weight;

 
  vector <fastjet::PseudoJet> list_pmed;
  vector <fastjet::PseudoJet> list_hmed;
  vector <fastjet::PseudoJet> list_pvac;
  vector <fastjet::PseudoJet> list_hvac;

  TFile *file = TFile::Open("aa_disthist_0bkg.root","recreate");
  TH1F* h1 = new TH1F("h1",   "chi  spectrum hadron",                    20, 0., 1.);
  TH1F* h2 = new TH1F("h2",   "chi  spectrum w/ bkg",                    20, 0., 1.);
  TH1F* h3 = new TH1F("h3",   "number of constituents in medium hadron", 20, 0., 100.);
  TH1F* h4 = new TH1F("h4",   "number of constituents in vacuum hadron", 20, 0., 100.);
  TH1F* h5 = new TH1F("h5",   "number of constituents in medium w/bkg", 20, 0., 100.);
  TH1F* h6 = new TH1F("h6",   "number of constituents in vacuum hadron", 20, 0., 100.);
  TH1F* h7 = new TH1F("h7",   "p_{T} spectrum in medium hadron",         20, 0., 1600.);
  TH1F* h8 = new TH1F("h8",   "p_{T} spectrum in vacuum hadron",         20, 0., 1600.);
  TH1F* h9 = new TH1F("h9",   "p_{T} spectrum in medium w/bkg",         20, 0., 1600.);
  TH1F* h10 = new TH1F("h10", "p_{T} spectrum in vacuum",         20, 0., 100.);

  vector <TH1F*> hist_list = {h1, h2, h3, h4, h5, h6, h7, h8, h9, h10} ;

  int count = 0;

  bool bkg = true;
  double pTMin   = 5.0;
  double R = 0.4;
  double dRlim = 0.4;
  fastjet::JetDefinition jetDef(fastjet::genkt_algorithm , R, -1);
  fastjet::AreaDefinition areaDef(fastjet::active_area_explicit_ghosts, fastjet::GhostedAreaSpec(4.0, 1, 0.01));

  auto start = std::chrono::high_resolution_clock::now();

	my_file.open("/Users/idamarie/Physics/HYBRID_realistic.out", ios::in);
	if (!my_file) {
		cout << "No such file";
	}
	else {
    while ( std::getline(my_file, line)){//} && num_lines < 5000000 ) {

      if (isalpha(line[0]) || isalpha(line[3])){
        std::size_t found = line.find(str);
        if (found!=std::string::npos){
          //first
          lastevent = line;
        }
        std::size_t found3 = line.find(str3);
        if (found3!=std::string::npos){
          std::vector<std::string> weights;
          tokenize(line, delim, weights);
          weight = stod(weights[1]);
        }

        std::size_t found2 = line.find(str2);
        if (found2!=std::string::npos){
          //end

          vector <fastjet::PseudoJet> inclusiveJets1, sortedJetsV;
          fastjet::ClusterSequence clustSeq1(list_hvac, jetDef);
          inclusiveJets1 = clustSeq1.inclusive_jets(pTMin);
          sortedJetsV    = sorted_by_pt(inclusiveJets1); //Vacuum

          vector <fastjet::PseudoJet> inclusiveJets2, sortedJetsM;
          fastjet::ClusterSequence clustSeq2(list_hmed, jetDef);
          inclusiveJets2 = clustSeq2.inclusive_jets(pTMin);
          sortedJetsM    = sorted_by_pt(inclusiveJets2); //Medium

          if (sortedJetsV[0].pt() < 100){
            continue;
          }

          int med_index = 0;
          if (sortedJetsV[0].pt() < sortedJetsM[0].pt()){
            med_index = 1;
            if (sortedJetsV[0].pt() < sortedJetsM[1].pt()){
              med_index =2;
              if (sortedJetsV[0].pt() < sortedJetsM[2].pt()){
                cout << "double oh nooo" << endl;
              }
            }
          } 
          
          //take the hardest medium and find the hardest vacuum that is dR<0.4 from the medium
          
          double mindR = 5.;
          int nr = 0;
          int same = 0;
          while (mindR>dRlim){
            if (nr == sortedJetsV.size()-1){
              //cout << "too high" <<  mindR<<endl;
              //if (mindR >1){
              //  cout << "not ok" << endl;
              //}
              break;
            }
            double dR = sortedJetsM[med_index].delta_R(sortedJetsV[nr]);
            //cout << dR << endl;
            if (dR < mindR && sortedJetsM[med_index].pt() < sortedJetsV[nr].pt() ){
              same = nr;
              mindR = dR;
            } else {
              ++nr;
            }
          }

          if (sortedJetsM[med_index].pt() / sortedJetsV[same].pt() > 1) {
            cout << "too large" << endl;
          }

          double chi = sortedJetsM[med_index].pt() / sortedJetsV[same].pt() ;

          vector <fastjet::PseudoJet> bgparticles = getbackground(0, 0.63, 100, 77.6595);//Max 5000 bkg particles (1000 in each eta bin)
          //for (int i = 0; i<10;++i){ 
          //  cout << bgparticles[i].pt() << endl; 
          //}
          //cout << bgparticles.size() << endl;

          vector <fastjet::PseudoJet> allInputs = list_hmed;
          allInputs.insert(allInputs.end(), bgparticles.begin(), bgparticles.end());
          //cout << allInputs.size() << endl;
          //Medium
          vector <fastjet::PseudoJet> inclusiveJetsbkg, sortedJetsbkg;
          fastjet::ClusterSequence clustSeqbkg(allInputs, jetDef);
          inclusiveJetsbkg = clustSeqbkg.inclusive_jets(pTMin);
          sortedJetsbkg    = sorted_by_pt(inclusiveJetsbkg);

          //cout << sortedJetsbkg[0].constituents().size() << endl;
          mindR = 5.;
          nr = 0;
          int same1 = 0;
          while (mindR>dRlim){
            if (nr == sortedJetsbkg.size()-1){
              break;
            }
            double dR = sortedJetsM[med_index].delta_R(sortedJetsbkg[nr]);
            if (dR < mindR){
              same1 = nr;
              mindR = dR;
            } else {
              ++nr;
            }
          }
          //Vacuum
          /* 
          vector <fastjet::PseudoJet> notallInputs = list_hvac;
          notallInputs.insert(notallInputs.end(), bgparticles.begin(), bgparticles.end());
          vector <fastjet::PseudoJet> inclusiveJetsbkgV, sortedJetsbkgV;
          fastjet::ClusterSequenceArea clustSeqbkgV(notallInputs, jetDef, areaDef);
          inclusiveJetsbkgV = clustSeqbkgV.inclusive_jets(pTMin);
          sortedJetsbkgV    = sorted_by_pt(inclusiveJetsbkgV);

          mindR = 5.;
          nr = 0;
          int same0 = 0;
          while (mindR>dRlim){
            if (nr == sortedJetsbkgV.size()-1){
              break;
            }
            double dR = sortedJetsV[same].delta_R(sortedJetsbkgV[nr]);
            if (dR < mindR){
              same0 = nr;
              mindR = dR;
            } else {
              ++nr;
            }
          }

          chi = sortedJetsbkg[same1].pt()/sortedJetsbkgV[same0].pt();
          if (chi>1){
            cout << "help" << endl;
            cout << chi << endl;
            cout << sortedJetsM[med_index].pt()/ sortedJetsV[same].pt() << endl;
            cout << sortedJetsV[same].pt() << "#" << sortedJetsbkgV[same0].pt() << endl;
            cout << sortedJetsM[med_index].pt() << "#" << sortedJetsbkg[same1].pt() << endl;
            cout << sortedJetsM[med_index].delta_R(sortedJetsV[same]) << endl;
            cout << sortedJetsbkg[same1].delta_R(sortedJetsbkgV[same0]) << endl;
            cout << mindR << endl;
            cout << sortedJetsbkgV[0].pt() << "#" << sortedJetsbkgV[1].pt() << endl;
            cout << sortedJetsbkg[0].pt() << "#" << sortedJetsbkg[1].pt() << endl;
            cout << sortedJetsbkgV[0].delta_R(sortedJetsbkg[0]) << "#" << sortedJetsbkgV[0].delta_R(sortedJetsbkg[1]) << endl;
          }
          */

          //if (sortedJetsbkg[same1].pt() / sortedJetsV[same].pt() > 1) {
          //  cout << "too large w bkg" << endl;
          //  cout << sortedJetsbkg[same1].pt() << endl;
          //  cout << sortedJetsM[med_index].pt() << endl;
          //  cout << sortedJetsV[same].pt() << endl;
          //}

          double nr1 =sortedJetsbkg[same1].constituents().size();
          double nr2 = sortedJetsV[same].constituents().size();

          //cout << nr1-nr2 << endl;
          //cout << endl;


          json j2 = {{{"chi", chi},{"pt", sortedJetsbkg[same1].pt()}, {"pt_v", sortedJetsV[same].pt()}, {"nr", nr1}, {"nr_v", nr2}, {"w", weight} }};
          //cout << "p" << endl;
          for (int j=0; j<sortedJetsbkg[same1].constituents().size(); j++){
            //cout << "1" << endl;
            //if (sortedJetsbkg[same1].constituents()[j].pt() < 0.00001){
            //  cout << "$" << sortedJetsbkg[same1].constituents()[j].pt() << endl;
            //}
            j2.push_back({{"E",sortedJetsbkg[same1].constituents()[j].E()}, {"px", sortedJetsbkg[same1].constituents()[j].px()}, {"py", sortedJetsbkg[same1].constituents()[j].py()}, {"pz",sortedJetsbkg[same1].constituents()[j].pz()}});
          }

          o1 << j2<< "\n"; 

          //histograms for bkg
          h1->Fill(sortedJetsbkg[same1].pt()/sortedJetsV[same].pt());
          h3->Fill(nr1);
          h4->Fill(nr2);
          h7->Fill(sortedJetsbkg[same1].pt());
          h8->Fill(sortedJetsV[same].pt()); 

          
          //Background subtraction
          //for (int i= 0; i<list_hmed.size(); ++i){
          //  if (list_hmed[i].pt() < 0.00001){
          //    cout << "# " << allInputs[i].pt() << endl;
          //  }
          //}
          


          vector <fastjet::PseudoJet> inclusiveJetsbkgsub, sortedJetsbkgsub;
          fastjet::ClusterSequenceArea clustSeqbkgsub(allInputs, jetDef, areaDef);
          inclusiveJetsbkgsub = clustSeqbkgsub.inclusive_jets(pTMin);
          sortedJetsbkgsub    = sorted_by_pt(inclusiveJetsbkgsub);

          fastjet::JetDefinition jet_def_for_rho(fastjet::kt_algorithm, 0.4);
          fastjet::Selector rho_range =  fastjet::SelectorAbsRapMax(3.0);
          fastjet::JetMedianBackgroundEstimator bge_rho(rho_range, jet_def_for_rho, areaDef);
          fastjet::BackgroundJetScalarPtDensity *scalarPtDensity=new fastjet::BackgroundJetScalarPtDensity();
          //bge_rho.set_jet_density_class(scalarPtDensity);
          bge_rho.set_particles(allInputs);
          fastjet::contrib::ConstituentSubtractor subtractor(&bge_rho);

          fastjet::PseudoJet subtracted_jet = subtractor(sortedJetsbkgsub[same1]);

          nr1 = subtracted_jet.constituents().size();
          if (nr1==0.0){
            continue;
          }
          //if (subtracted_jet.pt() / sortedJetsV[same].pt() > 1) {
          //  cout << "too large w bkgsub" << endl;
          //}
          
          //cout << subtracted_jet.pt()/sortedJetsV[same].pt() << endl;
          //cout << chi << endl;
          

          json j3 = {{{"chi", chi},{"pt", subtracted_jet.pt()}, {"pt_v", sortedJetsV[same].pt()}, {"nr", nr1}, {"nr_v", nr2}, {"w", weight} }};
          for (int j=0; j<subtracted_jet.constituents().size(); j++){
            //cout << "1" << endl;
            //if (subtracted_jet.constituents()[j].pt() < 0.000001){
            //  cout << subtracted_jet.constituents()[j].pt() << endl;
            //  cout << subtracted_jet.constituents()[j].m2() << endl;
            //  cout << endl;
            //}
            j3.push_back({{"E",subtracted_jet.constituents()[j].E()}, {"px", subtracted_jet.constituents()[j].px()}, {"py", subtracted_jet.constituents()[j].py()}, {"pz",subtracted_jet.constituents()[j].pz()}});
          }

          o2 << j3<< "\n"; 

          //Histograms for bkg subtraction
          h2->Fill(subtracted_jet.pt()/sortedJetsV[same].pt());
          h5->Fill(nr1);
          h6->Fill(nr2);
          h9->Fill(subtracted_jet.pt()); 
          h10->Fill(sortedJetsV[same].pt());

          
          ++count;

          list_hmed.clear();
          list_hvac.clear();
        }
      } else {
        std::vector<std::string> coords;
        tokenize(line, delim, coords);
        double E = sqrt(pow(stod(coords[0]), 2)+ pow(stod(coords[1]),2) + pow(stod(coords[2]),2) + pow(stod(coords[3]),2));
        fastjet::PseudoJet jet = fastjet::PseudoJet(stod(coords[0]), stod(coords[1]), stod(coords[2]), E);
        if (jet.pt() == 0) continue;
        if (stoi(coords[5])==0 || stoi(coords[5])==1) {
          list_hmed.push_back(jet);
        } else if (stoi(coords[5])==5) {
          list_hvac.push_back(jet);
        } 
      }
      num_lines++;
      
    }
    //Have lists of madium and vacuum particles for hadrons







	}
  o1 << std::endl;
  o2 << std::endl;


	my_file.close();

  vector <string> list_lab = {"#chi", "#chi", "Nr. constituents", "Nr. constituents","Nr. constituents", "Nr. constituents", "p_{T}", "p_{T}", "p_{T}", "p_{T}"};
  vector <string> list_canv = {"chiHadron", "chibkg", "ConstitHadronMedium", "ConstitPartonVacuum","ConstitbkgMedium", "ConstitHadronVacuum2.", "ptHadronMedium", "ptHadronVacuum", "ptMediumbkg", "ptHadronVacuum2." };
  for (int i = 0; i<hist_list.size(); ++i){
    TCanvas *c1 = new TCanvas(list_canv[i].c_str());
    hist_list[i]->SetStats(0);
    hist_list[i]->GetXaxis()->SetTitle(Form("%s", list_lab[i].c_str()));
    hist_list[i]->GetXaxis()->CenterTitle(true);
    hist_list[i]->GetYaxis()->SetTitle("A.U.");
    hist_list[i]->GetYaxis()->SetTitleOffset(2.0);
    hist_list[i]->GetYaxis()->CenterTitle(true);
    hist_list[i]->SetFillColor(kBlue);
    //h1->SetTitle(Form("#hat{p_{t,}}_{min}=500GeV, anti-k_{t}, R=0.7, test"));
    hist_list[i]->Draw("hist");
    c1->SetLeftMargin(0.2);
    gStyle->SetTitleFontSize(0.01);
    c1->Write();
  }

  

  delete file;

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = duration_cast<std::chrono::microseconds>(stop - start);
  cout << "Time it takes = " << duration.count()*pow(10,-6) << " s" << endl;   
  cout << "Total number of jets: " << count << endl;

  return 0;
}













/*

struct Point { double E, x, y, z; };

void to_json(json& j, const Point& p)
{
    j = {{"E", p.E},{"px", p.x}, {"py", p.y}, {"pz", p.z}};
}



int main() {
  // Number of events, generated and listed ones (for jets).
  int nEvent    = 600000;
  int nr_test =  50000;
  int nr_val =  100000;
  int nr_tot  =  500000;
 
  // Select common parameters for SlowJet and FastJet analyses.
  int    power   = -1;     // -1 = anti-kT; 0 = C/A; 1 = kT.
  double R       = 0.7;    // Jet size.
  double pTMin   = 5.0;    // Min jet pT.
  double etaMax  = 2.5;    // Pseudorapidity range of detector.
  int    select  = 2;      // Which particles are included?
  int    massSet = 2;      // Which mass are they assumed to have?

  int count = 0;

  // Generator. Shorthand for event.
  Pythia pythia;
  Event& event = pythia.event;

  //Pythia settings
  ostringstream pythiaset;
  pythiaset << "pythia_settings.cmd";
  pythia.readFile(pythiaset.str());
  pythia.init();

  // Set up FastJet jet finder.
  double ghost_area=0.01;
  fastjet::JetDefinition jetDef(fastjet::genkt_algorithm , R, power);
  fastjet::AreaDefinition areaDef(fastjet::active_area_explicit_ghosts, fastjet::GhostedAreaSpec(2.5, 1, ghost_area));
  
  std::vector <fastjet::PseudoJet> fjInputs; 

  TFile *file = TFile::Open("pt_spectr_qq.root","recreate");
  TH1F* h1 = new TH1F("h1", "p_{t}  spectrum", 50, 0., 2000.);
  TH1F* h2 = new TH1F("h1", "p_{t}  spectrum", 50, 0., 2000.);
  TH1F* h3 = new TH1F("h1", "p_{t}  spectrum", 50, 0., 2000.);

  
  
  std::ofstream o1("quark_test.json");
  std::ofstream o2("quark_val.json");
  std::ofstream o3("quark_train.json");
      
  auto start = std::chrono::high_resolution_clock::now();
  // Begin event loop. Generate event. Skip if error.
  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
    if (count == nr_tot) continue;
    if (!pythia.next()) continue;
    // Begin FastJet analysis: extract particles from event record.
    fjInputs.resize(0);
    for (int i = 0; i < event.size(); ++i) if (event[i].isFinal()) {

      // Require visible/charged particles inside detector.
      if      (select > 2 &&  event[i].isNeutral() ) continue;
      else if (select == 2 && !event[i].isVisible() ) continue;
      if (etaMax < 20. && abs(event[i].eta()) > etaMax) continue;

      // Create a PseudoJet from the complete Pythia particle.
      fastjet::PseudoJet particleTemp = event[i];

      // Store acceptable particles as input to Fastjet.
      fjInputs.push_back( particleTemp);
    }
    
    // Run Fastjet algorithm and sort jets in pT order.
    vector <fastjet::PseudoJet> inclusiveJets, sortedJets;
    fastjet::ClusterSequence clustSeq(fjInputs, jetDef);
    inclusiveJets = clustSeq.inclusive_jets(pTMin);
    sortedJets    = sorted_by_pt(inclusiveJets);

    //vector <fastjet::PseudoJet> particles = sortedJets[0].constituents();
    if (sortedJets.size()==0) continue;
    if (sortedJets[0].constituents().size() < 2) continue;
    //if (sortedJets[0].pt() < 500) continue;

    double w = pythia.info.weight();
    vector <fastjet::PseudoJet> cons = sortedJets[0].constituents();
    


   


    
    std::vector<Point> v;

    for (int i=0; i<sortedJets[0].constituents().size(); i++){
      Point pk = {sortedJets[0].constituents()[i].E(),sortedJets[0].constituents()[i].px(),sortedJets[0].constituents()[i].py(),sortedJets[0].constituents()[i].pz()};
      v.push_back(pk);
    }
    
    json j = v;
    
    if (count < nr_test) {
      o1 << j << "\n";
      h1->Fill(sortedJets[0].pt(),w);
    } else if (count < nr_val){
      o2 << j << "\n";
      h2->Fill(sortedJets[0].pt(),w);
    } else {
      o3 << j << "\n";
      h3->Fill(sortedJets[0].pt(),w);
    }

    if (count == nr_test-1) {
      cout << "test data done" << endl;
    } else if (count == nr_val-1) {
      cout << "validation data done" << endl;
    } else if (count == nr_tot-1) {
      cout << "train data done" << endl;
    }
    
    ++count;


  } 
  o1 << std::endl;
  o2 << std::endl;
  o3 << std::endl;


  
  TCanvas *c1 = new TCanvas("c1");
  h1->SetStats(0);
  h1->GetXaxis()->SetTitle("p_{T}");
  h1->GetXaxis()->CenterTitle(true);
  h1->GetYaxis()->SetTitle("#frac{1}{N_{tot}} #frac{dN}{dp_{t}}");
  h1->GetYaxis()->SetTitleOffset(2.0);
  h1->GetYaxis()->CenterTitle(true);
  h1->SetTitle(Form("#hat{p_{t,}}_{min}=500GeV, anti-k_{t}, R=0.7, test"));
  h1->Draw("hist");
  c1->SetLeftMargin(0.2);
  gStyle->SetTitleFontSize(0.01);
  c1->Write();

  TCanvas *c2 = new TCanvas("c2");
  h2->SetStats(0);
  h2->GetXaxis()->SetTitle("p_{T}");
  h2->GetXaxis()->CenterTitle(true);
  h2->GetYaxis()->SetTitle("#frac{1}{N_{tot}} #frac{dN}{dp_{t}}");
  h2->GetYaxis()->SetTitleOffset(2.0);
  h2->GetYaxis()->CenterTitle(true);
  h2->SetTitle(Form("#hat{p_{t,}}_{min}=500GeV, anti-k_{t}, R=0.7, val"));
  h2->Draw("hist");
  c2->SetLeftMargin(0.2);
  gStyle->SetTitleFontSize(0.01);
  c2->Write();

  TCanvas *c3 = new TCanvas("c3");
  h3->SetStats(0);
  h3->GetXaxis()->SetTitle("p_{T}");
  h3->GetXaxis()->CenterTitle(true);
  h3->GetYaxis()->SetTitle("#frac{1}{N_{tot}} #frac{dN}{dp_{t}}");
  h3->GetYaxis()->SetTitleOffset(2.0);
  h3->GetYaxis()->CenterTitle(true);
  h3->SetTitle(Form("#hat{p_{t,}}_{min}=500GeV, anti-k_{t}, R=0.7, train"));
  h3->Draw("hist");
  c3->SetLeftMargin(0.2);
  gStyle->SetTitleFontSize(0.01);
  c3->Write();


  delete file;

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = duration_cast<std::chrono::microseconds>(stop - start);
  cout << "Time it takes = " << duration.count()*pow(10,-6) << " s" << endl;   
  cout << "Total number of jets: " << count << endl;

  return 0;
}

*/