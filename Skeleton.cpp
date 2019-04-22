//=============================================================================================
// Computer Graphics Sample Program: GPU ray casting
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;          // pos of eye

	layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";
// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	struct Material {
		vec3 ka, kd, ks;
		float  shininess;
		vec3 F0;
		bool rough, reflective;
	};

	struct Light {
		vec3 direction;
		vec3 Le, La;
	};

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;	// material index
	};

	struct Ray {
		vec3 start, dir;
	};

	struct Sphere {
		vec3 center;
		float radius;
	};

	struct Triangle{
		vec3 p1,p2,p3;
		vec3 n;
	};



	const int nMaxObjects = 500;
	const int nMaxTriangles = 40;

	uniform vec3 wEye; 
	uniform Light light;     
	uniform Material materials[5];  // diffuse, specular, ambient ref
	uniform int nObjects;
	uniform Sphere objects[20];
	uniform int nTriangles;
	uniform int mirrorType;
	uniform Triangle triangles[nMaxTriangles];

	in  vec3 p;					// point on camera window corresponding to the pixel
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	Hit intersect(const Sphere object, const Ray ray) {
		Hit hit;
		hit.t = -1;
		vec3 dist = ray.start - object.center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0;
		float c = dot(dist, dist) - object.radius * object.radius;
		float discr = b * b - 4.0 * a * c;
		if (discr < 0)
		   return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0 / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0 / a;
		if (t1 <= 0)
		   return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - object.center) / object.radius;
		return hit;
	}

	Hit intersectWithTriangle(const Triangle triangle, const Ray ray){
		Hit hit;
		hit.mat = mirrorType;
		hit.t = -1.0;
		hit.t = dot((triangle.p1 - ray.start), triangle.n) / dot(ray.dir, triangle.n);
		if(hit.t < 0.0)
			return hit;
		hit.position  = ray.start + ray.dir * hit.t;
		hit.normal = triangle.n;
		if(dot(cross((triangle.p2 - triangle.p1) ,(hit.position - triangle.p1)), triangle.n) > 0.0 && 
		   dot(cross((triangle.p3 - triangle.p2) ,(hit.position - triangle.p2)), triangle.n) > 0.0 &&
		   dot(cross((triangle.p1 - triangle.p3) ,(hit.position - triangle.p3)), triangle.n) > 0.0)
			return hit;
		hit.t = -2.0;
		return hit;
}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		for (int o = 0; o < nObjects; o++) {
			Hit hit = intersect(objects[o], ray); //  hit.t < 0 if no intersection
		    hit.mat = o + 2;	 
		
			
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  
				bestHit = hit;
		
		}
			
		for(int i = 0; i < nTriangles; i++){
			Hit hit = intersectWithTriangle(triangles[i], ray);
						if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  
								bestHit = hit;
		}			




		if (dot(ray.dir, bestHit.normal) > 0) 
			bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (int o = 0; o < nObjects; o++) 
			if (intersect(objects[o], ray).t > 0)
			   return true; //  hit.t < 0 if no intersection
		 for (int o = 0; o < nObjects; o++) {
			if(intersectWithTriangle(triangles[o], ray).t > 0)
				return true;
		}
			return false;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}

	const float epsilon = 0.0001f;
	const int maxdepth = 16;

	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0.0, 0.0, 0.0);
		int n = 0;

		for(int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) {
				outRadiance += weight *  light.La;
				break;
				}
			if (materials[hit.mat].rough) {
				outRadiance += weight * materials[hit.mat].ka * light.La;
				Ray shadowRay;
				shadowRay.start = hit.position + hit.normal * epsilon;
				shadowRay.dir = light.direction;
				float cosTheta = dot(hit.normal, light.direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance += weight * light.Le * materials[hit.mat].kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light.direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0)
					 outRadiance += weight * light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
				}
				break;
			}

			if (materials[hit.mat].reflective) {
				weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
				//outRadiance += light.La * 0.09;
			} 
		}
		return outRadiance;
	}

	void main() {

		Ray ray;
		ray.start = wEye; 
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1); 
	}



)";

class Material {
protected:
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	bool rough, reflective;
public:
	
	void SetUniform(unsigned int shaderProg, int mat) {
		char buffer[256];
		sprintf(buffer, "materials[%d].ka", mat);
		ka.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].kd", mat);
		kd.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].ks", mat);
		ks.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].shininess", mat);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform material.shininess cannot be set\n");
		sprintf(buffer, "materials[%d].F0", mat);
		F0.SetUniform(shaderProg, buffer);

		sprintf(buffer, "materials[%d].rough", mat);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, rough ? 1 : 0); else printf("uniform material.rough cannot be set\n");
		sprintf(buffer, "materials[%d].reflective", mat);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, reflective ? 1 : 0); else printf("uniform material.reflective cannot be set\n");
	}
};

class RoughMaterial : public Material {
public:
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
	
	RoughMaterial(vec3 _ka, vec3 _kd, vec3 _ks, float _shininess) {
		ka = _ka;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}

};

class SmoothMaterial : public Material {
public:
	SmoothMaterial(vec3 _F0) {
		F0 = _F0;
		rough = false;
		reflective = true;
	}
};

vec3 calculateF0(vec3 n, vec3 k) {
	vec3 res;
	res.x = (powf(n.x - 1.0f, 2.0f) + powf(k.x, 2.0f)) / (powf(n.x + 1.0f, 2.0f) + powf(k.x, 2.0f));
	res.y = (powf(n.y - 1.0f, 2.0f) + powf(k.y, 2.0f)) / (powf(n.y + 1.0f, 2.0f) + powf(k.y, 2.0f));
	res.z = (powf(n.z - 1.0f, 2.0f) + powf(k.z, 2.0f)) / (powf(n.z + 1.0f, 2.0f) + powf(k.z, 2.0f));
	return res;
}

SmoothMaterial* Gold() {
	vec3 F0 = calculateF0(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f));
	return new SmoothMaterial(F0);
}

SmoothMaterial* Silver() {
	vec3 F0 = calculateF0(vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f));
	return new SmoothMaterial(F0);
}

enum mirrorType {
	GOLD = 0,
	SILVER = 1
};

struct Sphere {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius) { 
		center = _center; 
		radius = _radius; 
	}

	void SetUniform(unsigned int shaderProg, int o) {
		char buffer[256];
		sprintf(buffer, "objects[%d].center", o);
		center.SetUniform(shaderProg, buffer);
		sprintf(buffer, "objects[%d].radius", o);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, radius); else printf("uniform %s cannot be set\n", buffer);
	}
};

struct Triangle {
	vec3 p1, p2, p3;
	vec3 n;

	Triangle(const vec3& _p1, const vec3& _p2, const vec3& _p3) {
		p1 = _p1; p2 = _p2; p3 = _p3;
		n = normalize(cross((p2 - p1), (p3 - p1)));
		
	}

	void SetUniform(unsigned int shaderProg, int o) {
		char buffer[256];
		sprintf(buffer, "triangles[%d].p1", o);
		p1.SetUniform(shaderProg, buffer);
		sprintf(buffer, "triangles[%d].p2", o);
		p2.SetUniform(shaderProg, buffer);
		sprintf(buffer, "triangles[%d].p3", o);
		p3.SetUniform(shaderProg, buffer);
		sprintf(buffer, "triangles[%d].n", o);
		n.SetUniform(shaderProg, buffer);
	}

};

void setMirrorMaterial(mirrorType asd, unsigned int shaderProg) {
	int location = glGetUniformLocation(shaderProg, "mirrorType");
	if (location >= 0)
		glUniform1i(location, asd);
	else
		printf("uniform mirrorType cannot be set\n");
}


class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	vec3 getEye() {
		return eye;
	}
	
	void set(vec3 _eye, vec3 _lookat, vec3 vup, double _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tan(fov / 2);
		up = normalize(cross(w, right)) * f * tan(fov / 2);
	}
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
					eye.y,
					-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
	void SetUniform(unsigned int shaderProg) {
		eye.SetUniform(shaderProg, "wEye");
		lookat.SetUniform(shaderProg, "wLookAt");
		right.SetUniform(shaderProg, "wRight");
		up.SetUniform(shaderProg, "wUp");
	}

};

struct Light {
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
	void SetUniform(unsigned int shaderProg) {
		La.SetUniform(shaderProg, "light.La");
		Le.SetUniform(shaderProg, "light.Le");
		direction.SetUniform(shaderProg, "light.direction");
	}
};


float rnd() { return (float)rand() / RAND_MAX; }
int rndsign(){ 
	if (rnd() > 0.5f)
		return 1;
	return -1; 
}
GPUProgram gpuProgram; // vertex and fragment shaders

class Scene {
	std::vector<Sphere*> objects;
	std::vector<Light*> lights;
	Camera camera;
	std::vector<Material*> materials;
	std::vector<Triangle*> triangles;
public:
	Camera getCamera() {
		return camera;
	}
	
	void build() {
		vec3 eye = vec3(0.5, 0.5, 11);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0.5, 0.5, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		lights.push_back(new Light(vec3(-1, 2, 1), vec3(1, 1, 1), vec3(0.9, 0.9, 0.9)));

		vec3 kd(0.3f, 0.2f, 0.1f);
		vec3 ks(1, 1, 1);
		//materials.push_back(new RoughMaterial(kd, ks, 50));
		materials.push_back(Gold());
		materials.push_back(Silver());
		materials.push_back(new RoughMaterial(vec3(0.6, 0.6, 0.4), vec3(0.1, 0.8, 0.3), vec3(1.0, 0.5, 0.5) , 100));
		materials.push_back(new RoughMaterial(vec3(0.7, 0.5, 0.3), vec3(0.8, 0.1, 0.3), vec3(0.5, 1.0, 0.5), 100));
		materials.push_back(new RoughMaterial(vec3(0.6, 0.6, 0.4), vec3(0.3, 0.1, 0.8), vec3(0.5, 1.0, 0.5), 100));
		objects.push_back(new Sphere(vec3(0.4, 0.3, 0),  0.15));
		objects.push_back(new Sphere(vec3(0.6, 0.65, 0), 0.15));
		objects.push_back(new Sphere(vec3(0.8, 0.35, 0), 0.15));
		setMirrorMaterial(GOLD, gpuProgram.getId());
	}
	void SetUniform(unsigned int shaderProg) {
		int location = glGetUniformLocation(shaderProg, "nObjects");
		if (location >= 0) 
			glUniform1i(location, objects.size());
		else 
			printf("uniform nObjects cannot be set\n");

		
		for (int i = 0; i < objects.size(); i++)
			objects[i]->SetUniform(shaderProg, i);

		lights[0]->SetUniform(shaderProg);
		camera.SetUniform(shaderProg);

		for (int mat = 0; mat < materials.size(); mat++)
			materials[mat]->SetUniform(shaderProg, mat);
	}
	
	void Animate(float dt) {
		camera.Animate(dt);
	}

	void doBrown(unsigned int shaderProg) {
		const float d = 150.0f;
		for (int i = 0; i < objects.size(); i++) {
			Sphere* tmp = objects[i];
			vec3 randomvec = vec3(rndsign() * rnd() / d, rndsign() *  rnd() / d, rndsign() *  rnd() / d);
			if (validMove(tmp, randomvec)) {
				tmp->center = tmp->center + randomvec;
				char buffer[128];
				sprintf(buffer, "objects[%d].center", i);
				tmp->center.SetUniform(shaderProg, buffer);
			}
		}
	}

	bool validMove(const Sphere* tmp, const vec3 randomvec) {
		vec3 newcentre = tmp->center + randomvec;
		for(int i = 0; i < objects.size(); i++)
			if (objects[i] != tmp) 
				if (length(newcentre - objects[i]->center) < max(tmp->radius, objects[i]->radius))
					return false;
		if (isInsideBoundaris(newcentre, tmp->radius))
			return true;
		return false;
	}

	bool isInsideBoundaris(vec3 centre, float radius) {
		return ((centre.x + radius < 1.1f && centre.x - radius > -0.1f) &&
				(centre.y + radius < 1.1f && centre.y - radius > -0.1f) &&
				(centre.z + radius < 0.4f && centre.z - radius > -0.6f));
	}
};

Scene scene;
float rotation = 0;

class MirrorSystemManager {
	int n = 3;
	const float r = 0.55f;
	std::vector<Triangle*> triangles;
	const float rotationScale = 2 * M_PI / 360.0f;
	void deleteTriangles() {
		if (!triangles.empty()) {
			for (int i = 0; i < triangles.size(); i++)
				delete triangles[i];
			triangles.clear();
		}
	}

public:
	void build() {
		deleteTriangles();
		const int minN = 3;
		const int maxN = 10;
		const float z0 = 0.5f;
		const float z1 = scene.getCamera().getEye().z;
		const vec2 centre = vec2(0.5f, 0.5f);
		const float offset =  (2 * M_PI / n);
		for (int i = 0; i < n; i++) {
			const float x0 = cosf(rotation + i*offset)*r + centre.x;
			const float x1 = x0;
			const float x2 = cosf(rotation + (i+1)*offset)*r + centre.x;
			const float y0 = sinf(rotation + i*offset)*r + centre.y;
			const float y1 = y0;
			const float y2 = sinf(rotation + (i + 1)*offset)*r + centre.y;
			triangles.push_back(new Triangle(vec3(x0, y0, z0), vec3(x1, y1, z1), vec3(x2, y2, z1)));
			triangles.push_back(new Triangle(vec3(x2, y2, z0), vec3(x0, y0, z0), vec3(x2, y2, z1)));
		}

	}
	void increaseN() {
		if (n < 20) {
			n++;
			build();
		}
	}

	void decreaseN() {
		if (n > 3) {
			n--;
			build();
		}
	}

	void SetUniform(unsigned int shaderProg) {
		int location = glGetUniformLocation(shaderProg, "nTriangles");
		if (location >= 0)
			glUniform1i(location, triangles.size());
		else
			printf("uniform nTriangles cannot be set\n");

		for(int i = 0; i < triangles.size(); i++)
			triangles[i]->SetUniform(shaderProg, i);
	}
};



class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
public:
	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;
MirrorSystemManager mrs;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	gpuProgram.Create(vertexSource, fragmentSource, "fragmentColor");
	scene.build();
	mrs.build();
	fullScreenTexturedQuad.Create();

	gpuProgram.Use();
}

bool rotate = false;

void onDisplay() {
	static int nFrames = 0;
	nFrames++;
	static long tStart = glutGet(GLUT_ELAPSED_TIME);
	long tEnd = glutGet(GLUT_ELAPSED_TIME);

	glClearColor(1.0f, 0.5f, 0.8f, 1.0f);							
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
	scene.SetUniform(gpuProgram.getId());
	mrs.SetUniform(gpuProgram.getId());
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();								
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'r') {
		rotate = !rotate;
	}
	if (key == 'a') {
		mrs.increaseN();
	}
	if (key == 'd') {
		mrs.decreaseN();
	}
	if (key == 'g') {
		setMirrorMaterial(GOLD, gpuProgram.getId());
	}
	if (key == 's') {
		setMirrorMaterial(SILVER, gpuProgram.getId());
	}
	if (key == 54) { //number six on numpad
		rotation += 2 * M_PI / 360.0f;
		mrs.build();
	}
	if (key == 52) { //number four on numpad
		rotation -= 2 * M_PI / 360.0f;
		mrs.build();
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}
void onMouseMotion(int pX, int pY) {}

void onIdle() {
	if(rotate)
		scene.Animate(0.01);
	scene.doBrown(gpuProgram.getId());
	glutPostRedisplay();
}
