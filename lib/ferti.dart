import 'package:flutter/material.dart';
import 'package:floating_navigation_bar/floating_navigation_bar.dart';
import 'package:flutter1/main.dart';
import 'package:flutter1/recommendation.dart';
import 'package:get/get.dart';
import 'package:flutter1/Lab.dart';
import 'package:flutter1/Setting.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';


const String apiUrl = 'https://fertilizermodeldeployment.azurewebsites.net/';

void main() {
  runApp(const MaterialApp(
    title: 'Agriguru',
    home: MyApp(),
  ));
}

class MyApp extends StatefulWidget {
  const MyApp();
  @override
  RecommendedCropsPage createState() => RecommendedCropsPage();
}

class RecommendedCropsPage extends State<MyApp> {

  int soilInt = 0;
  int cropInt = 0;
  String ferti1='Loading...';
  String ferti2='Loading...';
  String ferti3='Loading...';
  String ferti4='Loading...';
  String ferti5='Loading...';

  String fertilizer1="loading...";
  String fertilizer2="loading...";
  String fertilizer3="loading...";
  String fertilizer4="loading...";
  String fertilizer5="loading...";


  int moisture = 22;
  int ni = 0;
  int pi = 0;
  int ki = 0;
  int tempi = 0;
  int humidi = 0;

  List<String> items = [
    'Soil Type',
    'Sandy',
    'Loamy',
    'Black',
    'Red',
    'Clayey'
  ];
  String dropdownValue = 'Soil Type';

  List<String> items2 = [
    'Crop Type',
    'MAIZE',
    'KIDNEYBEANS',
    'COTTON',
    'MANGO',
    'PIGEONPEAS',
    'MOTHBEANS',
    'MUNGBEAN',
    'BLACKGRAM',
    'LENTIL',
    'POMEGRANATE',
    'BANANA'
  ];
  String dropdownValue2 = 'Crop Type';

  Future<void> getData() async {
    try{
      SharedPreferences prefs = await SharedPreferences.getInstance();
      double n = prefs.getDouble('n') ?? 0.0;
      double p = prefs.getDouble('p') ?? 0.0;
      double k = prefs.getDouble('k') ?? 0.0;
      double temp = prefs.getDouble('temp') ?? 0.0;
      double humid = prefs.getDouble('humid') ?? 0.0;
      moisture = 22;
      ni = n.toInt();
      pi = p.toInt();
      ki = k.toInt();
      tempi = temp.toInt();
      humidi = humid.toInt();


      print("$n+$p+$k+$temp+$humid");

      // Map<String, int> soilTypes = {
      //   'Black': 0,
      //   'Clayey': 1,
      //   'Loamy': 2,
      //   'Red': 3,
      //   'Sandy': 4
      // };
      // Map<String, int> crops = {
      //   'COTTON': 0,
      //   'MAIZE': 1,
      //   'BANANA': 2,
      //   'BLACKGRAM': 3,
      //   'KIDNEYBEANS': 4,
      //   'LENTIL': 5,
      //   'MANGO': 6,
      //   'MOTHBEANS': 7,
      //   'MUNGBEAN': 8,
      //   'PIGEONPEAS': 9,
      //   'POMEGRANATE': 10,
      // };


    }catch (e) {
      throw Exception('lmoa No Data Found');
    }
  }

  Future<String> getFertiPredictions(int n, int p, int k, int temp, int humid, int moisture, int soiltype,int croptype) async {
    try {
      String res="";
      final responsa = await http.get(Uri.parse("$apiUrl?temp=$temp&humid=$humid&moist=$moisture&soil=$soiltype&crop=$croptype&n=$n&p=$p&k=$k")).then((value) {
        print("nuking successfull");
        print(value.statusCode);
        if (value.statusCode == 200) {
          print("siddhu ded");
          Map<String, dynamic> data = json.decode(value.body);
          print("worked ferti");
          print(data);
          res=data['prediction'];
          print(res);
          dynamic l="$apiUrl?temp=$temp&humid=$humid&moist=$moisture&soil=$soiltype&crop=$croptype&n=$n&p=$p&k=$k";
          print("url is: $l");
        } else {
          throw Exception('Failed to load crop predictions');
        }
      }).onError((error, stackTrace) {
        print("tf is nuke at");
      });
      return res;
    } catch (e) {
      dynamic l="$apiUrl?temp=$temp&humid=$humid&moist=$moisture&soil=$soiltype&crop=$croptype&n=$n&p=$p&k=$k";
      throw Exception('Failed to connect to server: $e with url $l');
    }
  }


  @override
  void initState(){
    // getData();
    setState(() {
    });
  }
  Widget build(BuildContext context) {

    double displayWidth = MediaQuery.of(context).size.width;
    double displayHeight = MediaQuery.of(context).size.height;
    var currentIndex = 0;
    return Scaffold(
        body: Column(
          children: [
            // Row(
            //   mainAxisAlignment: MainAxisAlignment.start,
            //   children: [
            //     Container(
            //       width: 100,
            //       margin: EdgeInsets.only(left: 25),
            //       child: InputDecorator(
            //         decoration: InputDecoration(
            //             border: InputBorder.none,
            //             contentPadding: EdgeInsets.all(5)),
            //         child: DropdownButtonHideUnderline(
            //           child: DropdownButton<String>(
            //             value: dropdownValue,
            //             items: items.map<DropdownMenuItem<String>>((String value) {
            //               return DropdownMenuItem<String>(
            //                   value: value, child: Text(value));
            //             }).toList(),
            //             onChanged: (String? newValue) {
            //               setState(
            //                     () {
            //                   dropdownValue = newValue ?? '';
            //                 },
            //               );
            //             },
            //           ),
            //         ),
            //       ),
            //     ),
            //     SizedBox(width: 75),
            //     Container(
            //       width: 160,
            //       child: InputDecorator(
            //         decoration: InputDecoration(
            //             border: InputBorder.none,
            //             contentPadding: EdgeInsets.only(right: 10)),
            //         child: DropdownButtonHideUnderline(
            //           child: DropdownButton<String>(
            //             value: dropdownValue2,
            //             items: items2.map<DropdownMenuItem<String>>((String value) {
            //               return DropdownMenuItem<String>(
            //                   value: value, child: Text(value));
            //             }).toList(),
            //             onChanged: (String? newValue) {
            //               setState(
            //                     () {
            //                   dropdownValue2 = newValue ?? '';
            //                 },
            //               );
            //             },
            //           ),
            //         ),
            //       ),
            //     ),
            //   ],
            // ),
            SizedBox(height: 10),
            ElevatedButton(onPressed: () async{

              print("beginning nuke sequence");
              print("nuke 1 send");
              ferti1 = await getFertiPredictions(ni, pi, ki, tempi, humidi, moisture, soilInt, cropInt);
              print("nuke 2 send");
              ferti2 = await getFertiPredictions(ni, pi, ki, tempi, humidi, moisture, soilInt, cropInt);
              print("nuke 3 send");
              ferti3 = await getFertiPredictions(ni, pi, ki, tempi, humidi, moisture, soilInt, cropInt);
              print("nuke 4 send");
              ferti4 = await getFertiPredictions(ni, pi, ki, tempi, humidi, moisture, soilInt, cropInt);
              print("nuke 5 send");
              ferti5 = await getFertiPredictions(ni, pi, ki, tempi, humidi, moisture, soilInt, cropInt);
              setState(() {
                fertilizer1='$ferti1';
                fertilizer2='$ferti2';
                fertilizer3='$ferti3';
                fertilizer4='$ferti4';
                fertilizer5='$ferti5';
              });

              },
                style: ElevatedButton.styleFrom(
                  primary: const Color(0xff1C9C73),
                ),
                child: Text('Predict Fertilizer')),
            Container(
              margin: EdgeInsets.only(right: 34),
              child: Text("recom_ferti".tr,
                  style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
            ),
            Expanded(
              child: ListView(
                children: [
                  SizedBox(height: 20),
                  Container(
                    padding: EdgeInsets.all(16),
                    child: Row(
                      children: [
                        Image.asset(
                          'assets/images/fertilizer1.png',
                          height: 100,
                        ),
                        SizedBox(width: 16),
                        Container(
                          padding: EdgeInsets.only(bottom: 30),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisAlignment: MainAxisAlignment.start,
                            children: [
                              Text(
                                fertilizer1,
                                style: TextStyle(
                                    fontWeight: FontWeight.bold, fontSize: 20),
                              ),
                              SizedBox(height: 8),
                              Text(
                                '',
                                style: TextStyle(
                                    fontSize: 12, fontWeight: FontWeight.bold),
                              ),
                              SizedBox(height: 5),
                              Text(
                                'Company 1',
                                style: TextStyle(
                                  color: Colors.grey,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding: EdgeInsets.all(16),
                    child: Row(
                      children: [
                        Image.asset(
                          'assets/images/fertilizer2.png',
                          height: 100,
                        ),
                        SizedBox(width: 16),
                        Container(
                          padding: EdgeInsets.only(bottom: 30),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisAlignment: MainAxisAlignment.start,
                            children: [
                              Text(
                                fertilizer2,
                                style: TextStyle(
                                    fontWeight: FontWeight.bold, fontSize: 20),
                              ),
                              SizedBox(height: 8),
                              Text(
                                '',
                                style: TextStyle(
                                    fontSize: 12, fontWeight: FontWeight.bold),
                              ),
                              SizedBox(height: 5),
                              Text(
                                'Company 2',
                                style: TextStyle(
                                  color: Colors.grey,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding: EdgeInsets.all(16),
                    child: Row(
                      children: [
                        Image.asset(
                          'assets/images/fertilizer3.png',
                          height: 100,
                        ),
                        SizedBox(width: 16),
                        Container(
                          padding: EdgeInsets.only(bottom: 30),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisAlignment: MainAxisAlignment.start,
                            children: [
                              Text(
                                fertilizer3,
                                style: TextStyle(
                                    fontWeight: FontWeight.bold, fontSize: 20),
                              ),
                              SizedBox(height: 8),
                              Text(
                                '',
                                style: TextStyle(
                                    fontSize: 12, fontWeight: FontWeight.bold),
                              ),
                              SizedBox(height: 5),
                              Text(
                                'Company 3',
                                style: TextStyle(
                                  color: Colors.grey,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding: EdgeInsets.all(16),
                    child: Row(
                      children: [
                        Image.asset(
                          'assets/images/fertilizer4.png',
                          height: 120,
                        ),
                        SizedBox(width: 16),
                        Container(
                          padding: EdgeInsets.only(bottom: 30),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisAlignment: MainAxisAlignment.start,
                            children: [
                              Text(
                                fertilizer4,
                                style: TextStyle(
                                    fontWeight: FontWeight.bold, fontSize: 20),
                              ),
                              SizedBox(height: 8),
                              Text(
                                '',
                                style: TextStyle(
                                    fontSize: 12, fontWeight: FontWeight.bold),
                              ),
                              SizedBox(height: 5),
                              Text(
                                'Company 4',
                                style: TextStyle(
                                  color: Colors.grey,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding: EdgeInsets.all(16),
                    child: Row(
                      children: [
                        Image.asset(
                          'assets/images/fertilizer5.png',
                          height: 100,
                        ),
                        SizedBox(width: 16),
                        Container(
                          padding: EdgeInsets.only(bottom: 30),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisAlignment: MainAxisAlignment.start,
                            children: [
                              Text(
                                fertilizer5,
                                style: TextStyle(
                                    fontWeight: FontWeight.bold, fontSize: 20),
                              ),
                              SizedBox(height: 8),
                              Text(
                                '',
                                style: TextStyle(
                                    fontSize: 12, fontWeight: FontWeight.bold),
                              ),
                              SizedBox(height: 5),
                              Text(
                                'Company 5',
                                style: TextStyle(
                                  color: Colors.grey,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            )
          ],
        ),

        extendBody: true,
        bottomNavigationBar: FloatingNavigationBar(
          backgroundColor: const Color(0xffD9D9D9),
          iconColor: Colors.black,
          barHeight: 70,
          textStyle: const TextStyle(
            color: Colors.black,
            fontSize: 14.0,
          ),
          indicatorColor: const Color(0xff1C9C73),
          indicatorWidth: 30,
          iconSize: 25.0,
          items: [
            NavBarItems(icon: Icons.home,title: "Nav_home".tr),
            NavBarItems(icon: Icons.addchart,title: "Nav_crop".tr),
            NavBarItems(icon: Icons.location_on, title: "Nav_lab".tr),
            NavBarItems(icon: Icons.settings_rounded,title: "Nav_setting".tr),
          ],
          onChanged: (value) {
            setState(() {
              currentIndex = value;
            });
            if(value==0){
              Navigator.push(context,
                  MaterialPageRoute(builder: (context)=>const HomePage())
              );
            }
            if(value==2){
              Navigator.push(context,
                  MaterialPageRoute(builder: (context)=>const Labpage())
              );
            }
            if(value==3){
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => SettingsPage()),
              );
            }
          }
      ),

    );

  }
}
