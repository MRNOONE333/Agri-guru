import 'package:flutter/material.dart';
import 'package:floating_navigation_bar/floating_navigation_bar.dart';
import 'package:flutter1/Setting.dart';
import 'package:flutter1/accountinfo.dart';
import 'package:flutter1/ferti.dart';
import 'package:flutter1/main.dart';
import 'package:flutter1/recommendation.dart';
import 'package:get/get.dart';
import 'package:flutter1/Lab.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';


const String apiUrl = 'http://18.197.37.111/';

void main() {
  runApp(
    const MaterialApp(
      title: 'AgriGuru',
      home: Recommendation(),
    ),
  );
}

class Recommendation extends StatefulWidget {
  const Recommendation();
  @override
  RecommendationState createState() => RecommendationState();
}

class RecommendationState extends State<Recommendation> {
  late int crop1Int;
  late int crop2Int;
  late int crop3Int;
  double price1=0.0;
  double price2=0.0;
  double price3=0.0;
  bool _Loading = true;

  String txt1="loading...";
  String txt2="loading...";
  String txt3="loading...";

  int state = 16; //haryana
  int district = 299; // gurgaon
  int market = 45; //patudi
  int month = 2;
  int season = 0;
  int day = 3;

  var currentIndex = 0;
  final TextEditingController _controller = TextEditingController();
  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
  final List<Color> boxColors = [
    const Color(0xff1c9c73),
    const Color(0xff9dd5df),
    const Color(0xffebb8af),
  ];

  final List<String> boxTexts = [
    'Wheat',
    'Rice',
    'Paddy',
  ];

  Future<void> getData() async {
    try{
      SharedPreferences prefs = await SharedPreferences.getInstance();
      String crop1 = prefs.getString('crop1') ?? '';
      String crop2 = prefs.getString('crop2') ?? '';
      String crop3 = prefs.getString('crop3') ?? '';
      Map<String, int> crops = {
        'rice': 156,
        'maize': 13,
        'chickpea': 18,
        'kidneybeans': 4,
        'pigeonpeas': 32,
        'mothbeans': 24,
        'mungbean': 45,
        'blackgram': 178,
        'lentil': 67,
        'pomegranate': 113,
        'banana': 132,
        'mango': 80,
        'grapes': 127,
        'watermelon': 104,
        'muskmelon': 31,
        'apple': 17,
        'orange': 160,
        'papaya': 153,
        'coconut': 76,
        'cotton': 155,
        'jute': 25,
        'coffee': 41,
      };
      crop1Int = crops[crop1]!;
      crop2Int = crops[crop2]!;
      crop3Int = crops[crop3]!;
      price1 = await getPricePredictions(crop1Int, state, district, market, month);
      price2 = await getPricePredictions(crop2Int, state, district, market, month);
      price3 = await getPricePredictions(crop3Int, state, district, market, month);
      print("price1: $price1 price2:$price2 price3:$price3");
      setState(() {
        txt1="Rs.$price1 per kg";
        txt2="Rs.$price2 per kg";
        txt3="Rs.$price3 per kg";
      });

    }catch (e) {
      throw Exception('No Data Found');
    }
  }


  Future<double> getPricePredictions(int crop1, int state, int district,int market,int month) async {
    try {
      final response = await http.get(Uri.parse("$apiUrl?com=$crop1&state=$state&district=$district&market=$market&month=$month"));
      if (response.statusCode == 200) {
        Map<String, dynamic> data = json.decode(response.body);
        print("worked");
        double pr = double.parse(data["price"].toStringAsFixed(2));
        return pr;
      } else {
        throw Exception('Failed to load crop predictions');
      }
    } catch (e) {
      dynamic l="$apiUrl?com=$crop1&state=$state&district=$district&market=$market&month=$month";
      throw Exception('Failed to connect to server: $e with url $l');
    }
  }

  @override
  void initState(){
    getData();
    setState(() {
      _Loading = false;
    });
  }

  @override
  Widget build(BuildContext context) {


    final todo =  ModalRoute.of(context)!.settings.arguments as Map<String, dynamic>;
    print(todo);
    double displayWidth = MediaQuery.of(context).size.width;
    double displayHeight = MediaQuery.of(context).size.height;
    return Scaffold(
      body:
      _Loading ? Center(child: CircularProgressIndicator())
      :
      SingleChildScrollView(
        child: Container(
          margin: const EdgeInsets.fromLTRB(20, 0, 20, 10),
          width: displayWidth * 1,
          child: Column(
            children: [
              SizedBox(
                height: displayHeight * 0.05,
              ),
              Padding(
                padding: EdgeInsets.only(left: 10),
                child: Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    'crop'.tr,
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 20,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 10),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  Expanded(
                    child: Container(
                      height: 80,
                      margin: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: boxColors[0],
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Center(
                        child: Text(
                          todo["crop1"],
                          style: const TextStyle(
                            fontSize: 16,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                  ),
                  Expanded(
                    child: Container(
                      height: 80,
                      margin: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: boxColors[1],
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Center(
                        child: Text(
                          todo["crop2"],
                          style: const TextStyle(
                            fontSize: 16,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                  ),
                  Expanded(
                    child: Container(
                      height: 80,
                      margin: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: boxColors[2],
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Center(
                        child: Text(
                          todo["crop3"],
                          style: const TextStyle(
                            fontSize: 16,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              Container(
                margin: const EdgeInsets.symmetric(horizontal: 10),
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: const Color(0xffECEEE6),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'price_crop'.tr,
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 18,
                        color: Colors.black,
                      ),
                    ),
                    const SizedBox(height: 20),
                    Container(
                      width: double.infinity,
                      height: 50,
                      decoration: BoxDecoration(
                        color: const Color(0xff1c9c73),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Row(
                        children: [
                          const SizedBox(width: 10),
                          Image.asset(
                            'assets/images/paddy.png',
                            width: 30,
                            height: 30,
                            fit: BoxFit.contain,
                          ),
                          // const SizedBox(width: 10),
                          Padding(
                            padding: EdgeInsets.only(left: 20, top: 8),
                            child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    todo["crop1"],
                                    style: TextStyle(
                                        fontSize: 12,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white),
                                  ),
                                  SizedBox(height: 4),
                                  Text(
                                    txt1,
                                    style: TextStyle(
                                        fontSize: 12,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white),
                                  )
                                ]),
                          ),
                          const Spacer(),
                          ElevatedButton(
                            onPressed: () {
                              int state = 16; //haryana
                              int district = 299; // gurgaon
                              int market = 45; //patudi
                              int month = 2;
                              int season = 0;
                              int day = 3;
                              int crop1 = crop1Int;
                              getPricePredictions(crop1, state, district, market, month);
                            },
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.fromLTRB(5, 10, 8, 10),
                              backgroundColor: const Color(0xffE59A54),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10),
                              ),
                            ),
                            child:  Text(
                                'check_ferti1'.tr,
                                style: TextStyle(
                                  color: Colors.white,
                                ),
                              ),
                            ),
                          const SizedBox(width: 10)
                        ],
                      ),
                    ),
                    const SizedBox(height: 20),
                    Container(
                      width: double.infinity,
                      height: 50,
                      decoration: BoxDecoration(
                        color: const Color(0xff9DD5DF),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Row(
                        children: [
                          const SizedBox(width: 10),
                          Image.asset(
                            'assets/images/wheat.png',
                            width: 30,
                            height: 30,
                            fit: BoxFit.contain,
                          ),
                          // const SizedBox(width: 10),
                          Padding(
                            padding: EdgeInsets.only(left: 20, top: 8),
                            child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    todo["crop2"],
                                    style: TextStyle(
                                        fontSize: 12,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white),
                                  ),
                                  SizedBox(height: 4),
                                  Text(
                                    txt2,
                                    style: TextStyle(
                                        fontSize: 12,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white),
                                  )
                                ]),
                          ),
                          const Spacer(),
                          ElevatedButton(
                            onPressed: () {
                              int state = 16; //haryana
                              int district = 299; // gurgaon
                              int market = 45; //patudi
                              int month = 2;
                              int crop1 = crop2Int;
                              getPricePredictions(crop1, state, district, market, month);
                            },
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.fromLTRB(5, 10, 8, 10),
                              backgroundColor: const Color(0xffE59A54),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10),
                              ),
                            ),
                            child: GestureDetector(
                              child: Text(
                                'check_ferti2'.tr,
                                style: TextStyle(
                                  color: Colors.white,
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 10)
                        ],
                      ),
                    ),
                    const SizedBox(height: 20),
                    Container(
                      width: double.infinity,
                      height: 50,
                      decoration: BoxDecoration(
                        color: const Color(0xffEBB8AF),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Row(
                        children: [
                          const SizedBox(width: 10),
                          Image.asset(
                            'assets/images/rice.png',
                            width: 30,
                            height: 30,
                            fit: BoxFit.contain,
                          ),
                          // const SizedBox(width: 10),
                          Padding(
                            padding: EdgeInsets.only(left: 20, top: 8),
                            child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    todo["crop3"],
                                    style: TextStyle(
                                        fontSize: 12,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white),
                                  ),
                                  SizedBox(height: 4),
                                  Text(
                                    txt3,
                                    style: TextStyle(
                                        fontSize: 12,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white),
                                  )
                                ]),
                          ),
                          const Spacer(),
                          ElevatedButton(
                            onPressed: () {
                              int state = 16; //haryana
                              int district = 299; // gurgaon
                              int market = 45; //patudi
                              int month = 2;
                              int season = 0;
                              int day = 3;
                              int crop1 = crop3Int;
                              getPricePredictions(crop1, state, district, market, month);
                            },
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.fromLTRB(5, 10, 8, 10),
                              backgroundColor: const Color(0xffE59A54),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10),
                              ),
                            ),
                            child: Text(
                              'check_ferti3'.tr,
                              style: TextStyle(
                                color: Colors.white,
                              ),
                            ),
                          ),
                          const SizedBox(width: 10)
                        ],
                      ),
                    ),
                  ],
                ),
              )
            ],
          ),
        ),
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
