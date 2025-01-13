import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter1/login.dart';
import 'package:get/get.dart';
import 'package:google_sign_in/google_sign_in.dart';

void main() {
  runApp(const SignUp());
}

final TextEditingController _emailController = TextEditingController();
final TextEditingController _passwordController = TextEditingController();
final TextEditingController _nameController = TextEditingController();

class SignUp extends StatelessWidget {
  const SignUp({Key? key}) : super(key: key);

  signUpWithGoogle(context) async{

    GoogleSignInAccount? googleUser = await GoogleSignIn().signIn();

    GoogleSignInAuthentication? googleAuth = await googleUser?.authentication;

    AuthCredential credential = GoogleAuthProvider.credential(
        accessToken: googleAuth?.accessToken,
        idToken: googleAuth?.idToken
    );

    UserCredential user = await FirebaseAuth.instance.signInWithCredential(credential);

    print(user.user?.displayName);

    if(user.user!=null){
      print("Signed In");
      Navigator.push(
        context,
        MaterialPageRoute(builder: (context) => const Login()),
      );
    }

  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Sign Up Page',
      home: Scaffold(
        backgroundColor: const Color(0xFFECEEE6),
        body: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(30.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 20),
                Center(
                  child: Center(
                    child: Text(
                      'all_one'.tr,
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Colors.black,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                Center(
                  child: Image.asset('assets/images/logo.png',
                    width: 250,
                  ),
                ),
                const SizedBox(height: 20),
                Text(
                  'sign_up'.tr,
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Colors.black,
                  ),
                ),
                const SizedBox(height: 20),
                TextFormField(
                  controller: _emailController,
                  decoration: InputDecoration(
                    labelText: 'email_label'.tr,
                    prefixIcon: const Icon(Icons.email),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                TextFormField(
                  controller: _nameController,
                  decoration: InputDecoration(
                    labelText: 'full_name'.tr,
                    prefixIcon: const Icon(Icons.person),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                TextFormField(
                  obscureText: true,
                  controller: _passwordController,
                  decoration: InputDecoration(
                    labelText: 'password'.tr,
                    prefixIcon: const Icon(Icons.lock),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                Text(
                  'agree_policy'.tr,
                  style: TextStyle(
                    color: Colors.black,
                  ),
                ),
                const SizedBox(height: 20),
                Center(
                  child: SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: () {
                        print(_emailController.text);
                        print(_passwordController.text);
                        print(_nameController.text);
                        FirebaseAuth.instance.createUserWithEmailAndPassword(email: _emailController.text,
                            password: _passwordController.text).then((value) {
                              print("Successfull Login");
                              Navigator.push(
                              context,
                              MaterialPageRoute(builder: (context) => const Login()),
                              );
                          }).onError((error, stackTrace){
                            print("Error ${error.toString()}");
                        });

                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xff4faa8d),
                        padding: const EdgeInsets.symmetric(vertical: 15),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                        ),
                      ),
                      child: Text('sign_up'.tr),
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                Center(
                  child: SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: () {
                        signUpWithGoogle(context);
                      },
                      style: ElevatedButton.styleFrom(

                        backgroundColor: const Color(0xffd3d3d3),
                        padding: const EdgeInsets.symmetric(vertical: 15),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10),
                          side: const BorderSide(
                            width: 0,
                            color: Color(0xffeceee6),
                          ),
                        ),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Image.asset(
                            'assets/images/google.png',
                            height: 30,

                          ),
                          const SizedBox(width: 10),
                          Text('gog_sign'.tr,
                            style: TextStyle(
                              color: Colors.black,
                            ),

                          ),

                        ],
                      ),
                    ),
                  ),
                ),
                const SizedBox(
                  height: 20,
                ),
                Center(
                  child:TextButton(
                      onPressed: (){
                        Navigator.push(
                          context,
                          MaterialPageRoute(builder: (context) => const Login()),
                        );
                      },
                      child:Text(
                        'join_before'.tr,
                        style: TextStyle(
                            color: Colors.blue
                        ),
                      )
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}