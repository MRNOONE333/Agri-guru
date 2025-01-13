import 'package:flutter/material.dart';
import 'package:get/get.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Recommended Crops',
      home: RecommendedCropsPage(),
    );
  }
}

class RecommendedCropsPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    double displayWidth = MediaQuery.of(context).size.width;
    double displayHeight = MediaQuery.of(context).size.height;
    return Scaffold(
        appBar: AppBar(
          toolbarHeight: displayHeight * 0.125,
          title: Center(
            child: Padding(
              padding: EdgeInsets.all(displayWidth * 0.01),
              child: Row(
                children: [
                  Image.asset(
                    'assets/images/logo.png',
                    width: displayWidth * 0.3,
                  ),
                  SizedBox(
                    width: displayWidth * 0.425,
                  ),
                  TextButton(
                    onPressed: () {},
                    child: const CircleAvatar(
                      backgroundImage: AssetImage('assets/images/avatar.jpg'),
                      radius: 25,
                    ),
                  )
                ],
              ),
            ),
          ),
          backgroundColor: const Color(0xFFFBFCFA),
          elevation: 0,
          automaticallyImplyLeading: false,
        ),
        body: Column(
          children: [
            SizedBox(height: 20),
            Container(
              margin: const EdgeInsets.fromLTRB(20, 0, 20, 10),
              width: displayWidth * 1,
              decoration: BoxDecoration(
                color: const Color(0xffECEEE6),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                children: [
                  Expanded(
                    flex: 2,
                    child: Container(
                      margin:
                      const EdgeInsets.only(left: 22, top: 5, bottom: 5),
                      padding: const EdgeInsets.fromLTRB(5, 5, 10, 5),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text(
                            'crop'.tr,
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  Expanded(
                    flex: 2,
                    child: Container(
                      margin: const EdgeInsets.only(
                          left: 22, top: 5, bottom: 5, right: 5),
                      padding: const EdgeInsets.fromLTRB(5, 5, 20, 5),
                      decoration: BoxDecoration(
                        color: const Color(0xffE59A54),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text(
                            'fertilizer'.tr,
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
            SizedBox(height: 10),
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
                          'assets/fertilizer1.png',
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
                                'Fertilizer 1',
                                style: TextStyle(
                                    fontWeight: FontWeight.bold, fontSize: 20),
                              ),
                              SizedBox(height: 8),
                              Text(
                                'Rs. 200/kg',
                                style: TextStyle(
                                    fontSize: 12, fontWeight: FontWeight.bold),
                              ),
                              SizedBox(height: 5),
                              Text(
                                'contains xxxx yyyy zzzz nutrients',
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
                          'assets/fertilizer2.png',
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
                                'Fertilizer 2',
                                style: TextStyle(
                                    fontWeight: FontWeight.bold, fontSize: 20),
                              ),
                              SizedBox(height: 8),
                              Text(
                                'Rs. 200/kg',
                                style: TextStyle(
                                    fontSize: 12, fontWeight: FontWeight.bold),
                              ),
                              SizedBox(height: 5),
                              Text(
                                'contains xxxx yyyy zzzz nutrients',
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
                          'assets/fertilizer3.png',
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
                                'Fertilizer 3',
                                style: TextStyle(
                                    fontWeight: FontWeight.bold, fontSize: 20),
                              ),
                              SizedBox(height: 8),
                              Text(
                                'Rs. 200/kg',
                                style: TextStyle(
                                    fontSize: 12, fontWeight: FontWeight.bold),
                              ),
                              SizedBox(height: 5),
                              Text(
                                'contains xxxx yyyy zzzz nutrients',
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
                          'assets/fertilizer4.png',
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
                                'Fertilizer 4',
                                style: TextStyle(
                                    fontWeight: FontWeight.bold, fontSize: 20),
                              ),
                              SizedBox(height: 8),
                              Text(
                                'Rs. 200/kg',
                                style: TextStyle(
                                    fontSize: 12, fontWeight: FontWeight.bold),
                              ),
                              SizedBox(height: 5),
                              Text(
                                'contains xxxx yyyy zzzz nutrients',
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
                          'assets/fertilizer5.png',
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
                                'Fertilizer 5',
                                style: TextStyle(
                                    fontWeight: FontWeight.bold, fontSize: 20),
                              ),
                              SizedBox(height: 8),
                              Text(
                                'Rs. 200/kg',
                                style: TextStyle(
                                    fontSize: 12, fontWeight: FontWeight.bold),
                              ),
                              SizedBox(height: 5),
                              Text(
                                'contains xxxx yyyy zzzz nutrients',
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
        ));
  }
}